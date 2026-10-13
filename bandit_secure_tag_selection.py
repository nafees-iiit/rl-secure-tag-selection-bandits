"""
Learning Secure Tag Selection WITHOUT Eavesdropper CSI: Multi-Armed Bandits
==========================================================================
Author : Mohammad Nafees (Ph.D. Scholar, ECE, MNNIT Allahabad)

Problem
-------
An RF source illuminates K backscatter tags located at different places. A
passive eavesdropper is hidden near some of them. Optimal tag selection (OTS)
needs the eavesdropper's channel in every time slot -- impossible, because a
passive eavesdropper never transmits.

Idea (Reinforcement Learning)
-----------------------------
Treat every tag as an "arm" of a slot machine. In each slot the controller picks
one tag and later learns only ONE bit: was the transmission secure
(secrecy capacity >= R_s) or not (e.g. from a periodic security audit).
Bandit algorithms (epsilon-greedy, UCB1, Thompson Sampling) learn which tag is
statistically safest by balancing EXPLORATION (try tags) and EXPLOITATION
(use the best so far). A discounted Thompson Sampling also ADAPTS when the
eavesdropper moves.
"""

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1. Scenario
# ------------------------------------------------------------------
SEED   = 42
K      = 5            # number of tags (arms)
M      = 2            # Nakagami-m fading for all links
SNR_DB = 15           # transmit SNR (dB)
BETA   = 0.5          # reflection coefficient
RS     = 0.5          # target secrecy rate (bits/s/Hz)
T      = 10000        # time slots
RUNS   = 200          # independent repetitions (results are averaged)
# Average channel gains: each tag sits at a different place
OMEGA_H = np.array([1.0, 0.8, 1.2, 0.9, 0.7])   # source -> tag
OMEGA_G = np.array([1.0, 0.9, 1.3, 0.8, 1.1])   # tag -> legitimate reader
OMEGA_F1 = np.array([0.4, 0.15, 1.2, 0.6, 0.9])  # tag -> eavesdropper (position 1)
OMEGA_F2 = np.array([0.4, 1.2, 0.15, 0.6, 0.9])  # tag -> eavesdropper after it MOVES
SNR = 10 ** (SNR_DB / 10)
rng = np.random.default_rng(SEED)


def draw_channels(omega_f):
    """One time slot, all runs, all tags: returns power gains h, g, f (RUNS x K)."""
    h = rng.gamma(M, OMEGA_H / M, size=(RUNS, K))
    g = rng.gamma(M, OMEGA_G / M, size=(RUNS, K))
    f = rng.gamma(M, omega_f / M, size=(RUNS, K))
    return h, g, f


def secure(h, g, f):
    """True where the tag's secrecy capacity reaches R_s (no outage)."""
    cs = np.log2(1 + SNR * BETA * h * g) - np.log2(1 + SNR * BETA * h * f)
    return cs >= RS


def success_prob(omega_f, n=400000):
    """Long-run probability that each tag is secure (used only for the oracle)."""
    loc = np.random.default_rng(7)
    h = loc.gamma(M, OMEGA_H / M, size=(n, K))
    g = loc.gamma(M, OMEGA_G / M, size=(n, K))
    f = loc.gamma(M, omega_f / M, size=(n, K))
    return secure(h, g, f).mean(axis=0)


P1, P2 = success_prob(OMEGA_F1), success_prob(OMEGA_F2)
print("Secure-transmission probability of each tag (unknown to the learner):")
print("  eavesdropper position 1:", np.round(P1, 3), "-> best tag", P1.argmax() + 1)
print("  eavesdropper position 2:", np.round(P2, 3), "-> best tag", P2.argmax() + 1)


# ------------------------------------------------------------------
# 2. Bandit algorithms (vectorised over RUNS)
# ------------------------------------------------------------------
class EpsilonGreedy:
    name = "Epsilon-greedy (ε=0.1)"
    def __init__(self): self.n = np.zeros((RUNS, K)); self.s = np.zeros((RUNS, K))
    def choose(self, t):
        mean = np.where(self.n > 0, self.s / np.maximum(self.n, 1), 1.0)   # untried = optimistic
        arm = mean.argmax(axis=1)
        explore = rng.random(RUNS) < 0.1
        arm[explore] = rng.integers(0, K, explore.sum())
        return arm
    def update(self, arm, r):
        idx = np.arange(RUNS); self.n[idx, arm] += 1; self.s[idx, arm] += r


class UCB1:
    name = "UCB1"
    def __init__(self): self.n = np.zeros((RUNS, K)); self.s = np.zeros((RUNS, K))
    def choose(self, t):
        if t < K: return np.full(RUNS, t)                      # try every tag once
        mean = self.s / self.n
        bonus = np.sqrt(2 * np.log(t + 1) / self.n)            # optimism for rarely tried tags
        return (mean + bonus).argmax(axis=1)
    def update(self, arm, r):
        idx = np.arange(RUNS); self.n[idx, arm] += 1; self.s[idx, arm] += r


class ThompsonSampling:
    name = "Thompson Sampling"
    def __init__(self, gamma=1.0):
        self.a = np.ones((RUNS, K)); self.b = np.ones((RUNS, K)); self.gamma = gamma
        if gamma < 1: self.name = f"Discounted Thompson (γ={gamma})"
    def choose(self, t):
        return rng.beta(self.a, self.b).argmax(axis=1)         # sample a belief, pick the best
    def update(self, arm, r):
        if self.gamma < 1:                                     # slowly forget old evidence
            self.a = 1 + self.gamma * (self.a - 1); self.b = 1 + self.gamma * (self.b - 1)
        idx = np.arange(RUNS); self.a[idx, arm] += r; self.b[idx, arm] += 1 - r


# ------------------------------------------------------------------
# 3. Simulation loop (all schemes see the SAME channels in each slot)
# ------------------------------------------------------------------
def simulate(moving_eve):
    learners = [EpsilonGreedy(), UCB1(), ThompsonSampling(), ThompsonSampling(gamma=0.995)]
    names = ["Genie OTS (instantaneous full CSI)", "Oracle best fixed tag (knows statistics)"] + \
            [l.name for l in learners] + ["SOTS (main link only)", "Random tag"]
    success = {n: np.zeros(T) for n in names}
    picks_ts = np.zeros((T, K))                                 # how often TS picks each tag
    for t in range(T):
        omega_f = OMEGA_F2 if (moving_eve and t >= T // 2) else OMEGA_F1
        p = P2 if (moving_eve and t >= T // 2) else P1
        h, g, f = draw_channels(omega_f)
        ok = secure(h, g, f)                                    # RUNS x K: which tags are secure now
        idx = np.arange(RUNS)
        success[names[0]][t] = ok.any(axis=1).mean()
        success[names[1]][t] = ok[:, p.argmax()].mean()
        for l in learners:
            arm = l.choose(t)
            r = ok[idx, arm].astype(float)                      # 1-bit feedback: secure or not
            l.update(arm, r)
            success[l.name][t] = r.mean()
            if l.name == "Thompson Sampling":
                picks_ts[t] = np.bincount(arm, minlength=K) / RUNS
        success["SOTS (main link only)"][t] = ok[idx, (h * g).argmax(axis=1)].mean()
        success["Random tag"][t] = ok[idx, rng.integers(0, K, RUNS)].mean()
    return names, success, picks_ts


t0 = time.time()
names, succ_s, picks_s = simulate(moving_eve=False)
names, succ_m, picks_m = simulate(moving_eve=True)
print(f"\nSimulated 2 scenarios x {T} slots x {RUNS} runs in {time.time()-t0:.0f} s")

# ------------------------------------------------------------------
# 4. Numbers
# ------------------------------------------------------------------
rows = []
for n in names:
    rows.append({"Scheme": n,
                 "SOP static (last 2000 slots)": 1 - succ_s[n][-2000:].mean(),
                 "SOP moving eve (2000 slots before move)": 1 - succ_m[n][T//2-2000:T//2].mean(),
                 "SOP moving eve (first 1000 slots AFTER move)": 1 - succ_m[n][T//2:T//2+1000].mean(),
                 "SOP moving eve (last 2000 slots)": 1 - succ_m[n][-2000:].mean()})
table = pd.DataFrame(rows)
table.to_csv("results/sop_summary.csv", index=False)
pd.set_option("display.width", 220)
print("\nSecrecy outage probability (lower is better):")
print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

oracle = succ_s[names[1]]
regret = {n: np.cumsum(oracle - succ_s[n]) for n in names[2:6]}
pd.DataFrame({"slot": np.arange(1, T + 1), **regret}).iloc[::50].to_csv("results/regret_static.csv", index=False)
print("\nCumulative regret after all slots (static eavesdropper, lower is better):")
for n, r in regret.items():
    print(f"  {n:32s}: {r[-1]:7.1f} lost secure transmissions")

# ------------------------------------------------------------------
# 5. Figures
# ------------------------------------------------------------------
style = {"Genie OTS (instantaneous full CSI)": ("k", "-"),
         "Oracle best fixed tag (knows statistics)": ("gray", "--"),
         "Epsilon-greedy (ε=0.1)": ("tab:orange", "-"), "UCB1": ("tab:green", "-"),
         "Thompson Sampling": ("tab:blue", "-"), "Discounted Thompson (γ=0.995)": ("tab:red", "-"),
         "SOTS (main link only)": ("tab:purple", ":"), "Random tag": ("tab:brown", ":")}

def smooth(x, w=200):
    return np.convolve(x, np.ones(w) / w, mode="valid")

for tag, succ, title, fname in [
        ("static", succ_s, "Static eavesdropper", "sop_vs_time_static.png"),
        ("moving", succ_m, "Eavesdropper MOVES at slot 5000", "sop_vs_time_moving.png")]:
    plt.figure(figsize=(8, 6))
    for n in names:
        c, ls = style[n]
        plt.plot(np.arange(len(smooth(succ[n]))) + 200, 1 - smooth(succ[n]), color=c, ls=ls, label=n, lw=1.6)
    if tag == "moving":
        plt.axvline(T // 2, color="k", lw=1, ls="--"); plt.text(T // 2 + 80, 0.62, "eavesdropper moves", fontsize=8)
    plt.yscale("log"); plt.xlabel("Time slot"); plt.ylabel("Secrecy outage probability (200-slot average)")
    plt.title(f"{title}: learning secure tag selection without eavesdropper CSI", fontsize=10)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)
    plt.tight_layout(); plt.savefig(f"results/{fname}", dpi=200); plt.close()

plt.figure(figsize=(7, 4.6))
for n, r in regret.items():
    c, ls = style[n]; plt.plot(np.arange(1, T + 1), r, color=c, ls=ls, label=n, lw=1.8)
plt.xscale("log"); plt.xlabel("Time slot (log scale)"); plt.ylabel("Cumulative regret")
plt.title("Regret vs the oracle best tag (static eavesdropper)")
plt.grid(True, which="both", alpha=0.3); plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig("results/regret_static.png", dpi=200); plt.close()

plt.figure(figsize=(8, 4.8))
sm = np.array([smooth(picks_m[:, k], 100) for k in range(K)])
plt.stackplot(np.arange(sm.shape[1]) + 100, sm, labels=[f"Tag {k+1}" for k in range(K)], alpha=0.85)
plt.axvline(T // 2, color="k", lw=1, ls="--")
plt.xlabel("Time slot"); plt.ylabel("Fraction of runs choosing the tag")
plt.title("Thompson Sampling: which tag it chooses over time (eavesdropper moves at 5000)", fontsize=10)
plt.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=5, frameon=False)
plt.ylim(0, 1); plt.margins(x=0)
plt.tight_layout(); plt.savefig("results/tag_choices_thompson.png", dpi=200); plt.close()
print("\nAll results saved in the 'results' folder.")
