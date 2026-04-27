"""
sim_benchmark.py  —  Three-condition benchmark simulation
==========================================================
Reproduces Figures 3–5 and Table 1 from the paper.

Usage
-----
  python sim_benchmark.py                  # paper seed (42), single run
  python sim_benchmark.py --seed 123       # different seed, single run
  python sim_benchmark.py --runs 200       # Monte-Carlo over 200 seeds
  python sim_benchmark.py --runs 200 --seed 0   # reproducible Monte-Carlo
  python sim_benchmark.py --no-figures     # print table only, skip plots

Conditions
----------
  A  No schedule   – random topic selection, 2 hrs/day, slow decay
  B  Static        – sequential topics, 4 hrs/day fixed, no priority
  C  Adaptive      – proposed system: priority function, dependency graph,
                     geometric capacity recovery, psychological reset

Fair-hours note
---------------
Condition C accumulates more total hours than B because geometric capacity
recovery grows its daily budget from 2 hrs toward 7 hrs. The Monte-Carlo
run reports this clearly so you can judge how much of C's advantage is
due to hours vs. scheduling logic.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ── Syllabus ──────────────────────────────────────────────────────────────────
TOPICS = [
    {"name": "Kinematics",          "W": 0.80, "D": 0.50, "prereqs": []},
    {"name": "Newton's Laws",       "W": 0.90, "D": 0.60, "prereqs": [0]},
    {"name": "Work Power Energy",   "W": 0.85, "D": 0.65, "prereqs": [1]},
    {"name": "Rotational Motion",   "W": 0.85, "D": 0.80, "prereqs": [2]},
    {"name": "Gravitation",         "W": 0.70, "D": 0.60, "prereqs": [1]},
    {"name": "Properties of Matter","W": 0.60, "D": 0.55, "prereqs": []},
    {"name": "Thermodynamics",      "W": 0.80, "D": 0.70, "prereqs": []},
    {"name": "Kinetic Theory",      "W": 0.70, "D": 0.65, "prereqs": [6]},
    {"name": "Waves",               "W": 0.75, "D": 0.60, "prereqs": []},
    {"name": "Optics",              "W": 0.85, "D": 0.75, "prereqs": [8]},
    {"name": "Electrostatics",      "W": 0.90, "D": 0.75, "prereqs": []},
    {"name": "Current Electricity", "W": 0.85, "D": 0.70, "prereqs": [10]},
    {"name": "Magnetism",           "W": 0.80, "D": 0.75, "prereqs": [11]},
    {"name": "EMI",                 "W": 0.75, "D": 0.80, "prereqs": [12]},
    {"name": "AC Circuits",         "W": 0.70, "D": 0.75, "prereqs": [13]},
    {"name": "EM Waves",            "W": 0.60, "D": 0.50, "prereqs": []},
    {"name": "Ray Optics",          "W": 0.80, "D": 0.70, "prereqs": [9]},
    {"name": "Wave Optics",         "W": 0.75, "D": 0.75, "prereqs": [8]},
    {"name": "Modern Physics",      "W": 0.85, "D": 0.80, "prereqs": [15]},
    {"name": "Semiconductor",       "W": 0.70, "D": 0.65, "prereqs": []},
]

N     = len(TOPICS)
W_arr = np.array([t["W"] for t in TOPICS])
D_arr = np.array([t["D"] for t in TOPICS])

# ── Simulation parameters ─────────────────────────────────────────────────────
DAYS = 180

# Compliance sequence probabilities
P_FULL    = 0.70   # probability of a full study day
P_PARTIAL = 0.15   # probability of a partial day  (else missed)
PARTIAL_LO, PARTIAL_HI = 0.3, 0.8  # fraction of scheduled hours on partial day

# Forced absence windows (0-indexed days, inclusive)
ABSENCE_WINDOWS = [(25, 27), (105, 107)]

# High-priority threshold (coverage metric)
HIGH_W_THRESHOLD  = 0.75
MASTERY_THRESHOLD = 0.40   # minimum mastery to count as "covered"

# Condition A parameters
A_HOURS_PER_DAY = 2.0
A_SCORE_LO, A_SCORE_HI = 0.38, 0.62
A_DECAY = 0.997

# Condition B parameters
B_HOURS_PER_DAY  = 4.0
B_SCORE_LO, B_SCORE_HI = 0.45, 0.78
B_DECAY = 0.995
B_ADVANCE_AFTER  = 5      # sessions before moving to next topic

# Condition C parameters
C_K0           = 2.0   # initial capacity (hrs/day)
C_K_TARGET     = 7.0   # target capacity
C_R_BASE       = 0.05  # base daily growth rate
C_DECAY        = 0.995
C_SCORE_LO, C_SCORE_HI = 0.45, 0.75
C_DONE_THRESH  = 0.65  # mastery at which topic is marked done
C_UNDONE_THRESH= 0.48  # mastery below which a done topic is re-opened
C_PREREQ_MIN   = 0.15  # minimum prereq mastery to unlock a topic
C_PROP_GAMMA   = 0.9   # priority propagation damping factor
C_PROP_LAMBDA  = 0.7   # edge interconnectedness weight
C_SESSION_LEN  = 1.5   # hrs per study slot
C_MIN_SLOT_HRS = 0.18  # discard slots shorter than this
C_NEGLECT_MARK = 0.20  # mastery at or below this → topic is neglected
C_NEGLECT_HRS  = 2.5   # capacity threshold before forcing neglected topics in
C_REVISION_EVERY = 5   # review done topics every N days
C_REVISION_N   = 2     # how many done topics to review each cycle
C_REV_SCORE_LO, C_REV_SCORE_HI = 0.75, 0.95  # score range for revision
C_RESET_DAYS   = 3     # consecutive missed days before psychological reset
C_STREAK_DAYS  = 7     # consecutive compliant days before r boost
C_R_BOOST      = 1.1   # multiplicative boost after streak

# ── Colours ───────────────────────────────────────────────────────────────────
COL = {"A": "#DC2626", "B": "#2563EB", "C": "#16A34A"}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.18, "grid.linestyle": "--",
    "axes.labelsize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "legend.fontsize": 11, "legend.framealpha": 0.92,
    "figure.facecolor": "#FFFFFF", "axes.facecolor": "#F9F9F9",
})

# ── Helpers ───────────────────────────────────────────────────────────────────
HIGH_W = W_arr >= HIGH_W_THRESHOLD

def weighted_mastery(M):
    """Weighted average mastery over high-priority topics."""
    return float(np.sum(W_arr[HIGH_W] * M[HIGH_W]) / np.sum(W_arr[HIGH_W]))

def coverage(M):
    """Percentage of high-priority topics above MASTERY_THRESHOLD."""
    idx = [i for i in range(N) if W_arr[i] >= HIGH_W_THRESHOLD]
    return sum(1 for i in idx if M[i] > MASTERY_THRESHOLD) / len(idx) * 100

def rolling14(series):
    return pd.Series(series).rolling(14, min_periods=1).mean().tolist()

# ── Compliance generator ──────────────────────────────────────────────────────
def gen_compliance():
    """
    Generate DAYS-length compliance sequence.
    Full day = 1.0, partial day ∈ (PARTIAL_LO, PARTIAL_HI), missed = 0.0.
    Forced absence windows override the random draw after generation so that
    the random state consumed is identical regardless of which days are forced
    absent (this keeps seed=42 reproducible with the paper's figures).
    """
    seq = []
    for _ in range(DAYS):
        r = np.random.random()
        if r < P_FULL:
            seq.append(1.0)
        elif r < P_FULL + P_PARTIAL:
            seq.append(float(np.random.uniform(PARTIAL_LO, PARTIAL_HI)))
        else:
            seq.append(0.0)
    for lo, hi in ABSENCE_WINDOWS:
        for d in range(lo, hi + 1):
            seq[d] = 0.0
    return seq

# ── Condition A — No schedule ─────────────────────────────────────────────────
def run_A(comp):
    """
    Condition A: no formal schedule.
    Topics chosen randomly (Dirichlet-weighted), fixed 2 hrs/day.
    """
    M    = np.full(N, 0.2)
    n    = np.zeros(N, int)
    mlog = []
    hlog = []

    for day in range(DAYS):
        c = comp[day]
        if c > 0:
            h = c * A_HOURS_PER_DAY
            weights = np.random.dirichlet(np.ones(N) * 0.3)
            t = np.random.choice(N, p=weights)
            s = np.random.uniform(A_SCORE_LO, A_SCORE_HI)
            n[t] += 1
            alpha = 1.0 / n[t]
            M[t]  = np.clip(M[t] + alpha * (s - M[t]), 0, 1)
            for j in range(N):
                if j != t:
                    M[j] *= A_DECAY
        else:
            h = 0.0
            M[:] = np.clip(A_DECAY * M, 0, 1)
        hlog.append(h)
        mlog.append(weighted_mastery(M))

    return {"M": M, "m": mlog, "tot": sum(hlog)}

# ── Condition B — Static schedule ─────────────────────────────────────────────
def run_B(comp):
    """
    Condition B: fixed sequential schedule, 4 hrs/day.
    Advances to next topic after B_ADVANCE_AFTER sessions.
    No priority function, dependency graph, or capacity recovery.
    """
    M    = np.full(N, 0.2)
    n    = np.zeros(N, int)
    ptr  = 0
    mlog = []
    hlog = []

    for day in range(DAYS):
        c = comp[day]
        if c > 0:
            h = c * B_HOURS_PER_DAY
            t = ptr % N
            s = np.random.uniform(B_SCORE_LO, B_SCORE_HI)
            n[t] += 1
            alpha = 1.0 / n[t]
            M[t]  = np.clip(M[t] + alpha * (s - M[t]), 0, 1)
            if n[t] >= B_ADVANCE_AFTER:
                ptr += 1
            for j in range(N):
                if j != t:
                    M[j] *= B_DECAY
        else:
            h = 0.0
            M[:] = np.clip(B_DECAY * M, 0, 1)
        hlog.append(h)
        mlog.append(weighted_mastery(M))

    return {"M": M, "m": mlog, "tot": sum(hlog)}

# ── Condition C — Proposed adaptive system ────────────────────────────────────
def run_C(comp):
    """
    Condition C: full adaptive system.
    - Priority function: P_i = W_i × D_i × (1 − m_i)
    - Dependency graph with priority propagation
    - Geometric capacity recovery: K(t) = min(K0·(1+r)^t, K_target)
    - Psychological reset after C_RESET_DAYS consecutive missed days
    - Periodic revision of completed topics
    """
    M    = np.full(N, 0.2)
    n    = np.zeros(N, int)
    done = np.zeros(N, bool)

    K    = C_K0
    r    = C_R_BASE
    r_prev = C_R_BASE

    consec_missed = 0
    streak_buf    = []   # last C_STREAK_DAYS daily discipline scores
    resets        = []
    mlog          = []
    hlog          = []

    for day in range(DAYS):
        c = comp[day]

        # ── Psychological reset ──────────────────────────────────────────────
        if consec_missed >= C_RESET_DAYS:
            resets.append(day)
            consec_missed = 0
            r = C_R_BASE

        # Theoretical capacity curve (used for scheduling and discipline calc)
        K_theoretical = min(C_K0 * (1 + C_R_BASE) ** day, C_K_TARGET)
        studied_topics = []

        if c > 0:
            h = c * K_theoretical
            consec_missed = 0

            # ── Priority scores ──────────────────────────────────────────────
            P = W_arr * D_arr * (1.0 - M)

            # Mark topics as done / re-open them
            for i in range(N):
                if M[i] >= C_DONE_THRESH:
                    done[i] = True
                if M[i] < C_UNDONE_THRESH and done[i]:
                    done[i] = False
                if done[i]:
                    P[i] = 0.0

            # Priority propagation through dependency graph
            P2 = P.copy()
            for j in range(N):
                for p in TOPICS[j]["prereqs"]:
                    P2[p] += C_PROP_GAMMA * C_PROP_LAMBDA * P[j]
            P = P2

            # Eligible topics: prereqs sufficiently mastered, not done
            eligible = [
                i for i in range(N)
                if all(M[p] >= C_PREREQ_MIN for p in TOPICS[i]["prereqs"])
                and not done[i]
            ]
            if not eligible:
                eligible = [i for i in range(N) if not done[i]]
            if not eligible:
                eligible = list(range(N))

            # Bring in neglected topics when capacity allows
            neglected = [i for i in range(N) if M[i] <= C_NEGLECT_MARK + 1e-6]
            if neglected and h > C_NEGLECT_HRS:
                eligible = list(set(eligible + neglected[:3]))

            # Select top topics by priority and allocate hours proportionally
            top_topics = sorted(eligible, key=lambda i: P[i], reverse=True)
            n_slots    = min(len(top_topics), max(2, int(h / C_SESSION_LEN)))
            top        = top_topics[:n_slots]
            pv         = np.array([P[i] + 1e-9 for i in top])
            pv        /= pv.sum()
            hours_each = pv * h

            for idx, t in enumerate(top):
                if hours_each[idx] < C_MIN_SLOT_HRS:
                    continue
                s = np.random.uniform(C_SCORE_LO, C_SCORE_HI)
                n[t] += 1
                alpha = 1.0 / n[t]
                M[t]  = np.clip(M[t] + alpha * (s - M[t]), 0, 1)
                studied_topics.append(t)

            # Periodic revision of completed topics
            done_list = [i for i in range(N) if done[i]]
            if day % C_REVISION_EVERY == 0 and done_list and h > 0:
                for rev in sorted(done_list, key=lambda i: M[i])[:C_REVISION_N]:
                    s = np.random.uniform(C_REV_SCORE_LO, C_REV_SCORE_HI)
                    n[rev] += 1
                    alpha   = 1.0 / n[rev]
                    M[rev]  = np.clip(M[rev] + alpha * (s - M[rev]), 0, 1)
                    studied_topics.append(rev)

            # ── Update growth rate based on compliance ───────────────────────
            D = np.clip(h / K_theoretical, 0, 1) if K_theoretical > 0 else 1.0
            if D >= 0.8:
                r = max(r_prev, C_R_BASE)
            elif D > 0:
                r_prev = r
                r = max(r / 2, C_R_BASE * 0.3)
            streak_buf.append(D)
            if len(streak_buf) > C_STREAK_DAYS:
                streak_buf.pop(0)
            if len(streak_buf) == C_STREAK_DAYS and all(d >= 0.8 for d in streak_buf):
                r *= C_R_BOOST

        else:
            h = 0.0
            consec_missed += 1
            r_prev = r
            r = C_R_BASE * 0.1
            streak_buf.append(0.0)
            if len(streak_buf) > C_STREAK_DAYS:
                streak_buf.pop(0)

        # ── Knowledge decay for unstudied topics ─────────────────────────────
        for j in range(N):
            if j not in studied_topics:
                M[j] *= C_DECAY
        M = np.clip(M, 0, 1)

        # Capacity smoothed average
        K = 0.8 * K + 0.2 * h

        hlog.append(h)
        mlog.append(weighted_mastery(M))

    return {"M": M, "m": mlog, "resets": resets, "tot": sum(hlog)}

# ── Single-run figures ────────────────────────────────────────────────────────
def save_figures(rA, rB, rC):
    dx = np.arange(DAYS)

    # Fig 3 — weighted mastery over time (14-day rolling average)
    fig3, ax3 = plt.subplots(figsize=(8, 3.5))
    ax3.plot(dx, rolling14(rA["m"]), color=COL["A"], ls="--", lw=2.0, alpha=0.9,
             label="Condition A — No Schedule")
    ax3.plot(dx, rolling14(rB["m"]), color=COL["B"], ls=":",  lw=2.0, alpha=0.9,
             label="Condition B — Static Schedule (4 hrs/day)")
    ax3.plot(dx, rolling14(rC["m"]), color=COL["C"], ls="-",  lw=2.4, alpha=0.95,
             label="Condition C — Adaptive System (Proposed)")
    for rd in rC["resets"]:
        ax3.axvline(rd, color=COL["C"], ls=":", lw=0.9, alpha=0.45)
        ax3.annotate("\u21ba", xy=(rd + 1, 0.03), fontsize=10, color=COL["C"], alpha=0.8)
    ax3.set_xlabel("Day")
    ax3.set_ylabel("Weighted Mastery Score (0\u20131)")
    ax3.set_xlim(0, DAYS)
    ax3.set_ylim(0, 1.0)
    ax3.legend(loc="lower right", fontsize=11, framealpha=0.9)
    fig3.tight_layout(pad=0.6)
    fig3.savefig("weighted_mastery.png", dpi=300, bbox_inches="tight")
    plt.close(fig3)
    print("Saved weighted_mastery.png")

    cA = coverage(rA["M"])
    cB = coverage(rB["M"])
    cC = coverage(rC["M"])

    # Fig 4 — coverage bar chart
    fig4, ax4 = plt.subplots(figsize=(5, 4))
    bars = ax4.bar(
        ["Cond. A\n(No Schedule)", "Cond. B\n(Static, 4 hrs)", "Cond. C\n(Adaptive)"],
        [cA, cB, cC],
        color=[COL["A"], COL["B"], COL["C"]],
        edgecolor="white", linewidth=1.5, width=0.52, alpha=0.92,
    )
    for bar, val in zip(bars, [cA, cB, cC]):
        ax4.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            f"{val:.0f}%", ha="center", va="bottom",
            fontsize=13, fontweight="bold", color="#1F2937",
        )
    ax4.set_ylabel("High-Priority Coverage (%)")
    ax4.set_ylim(0, 115)
    fig4.tight_layout(pad=0.6)
    fig4.savefig("coverage_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig4)
    print("Saved coverage_bar.png")

    # Fig 5 — per-topic mastery sorted by weightage
    sort_idx = np.argsort(W_arr)[::-1]
    labs = [TOPICS[i]["name"] for i in sort_idx]
    fig5, ax5 = plt.subplots(figsize=(14, 4))
    x = np.arange(N)
    w = 0.28
    ax5.bar(x - w, rA["M"][sort_idx], w, color=COL["A"], alpha=0.85, edgecolor="white", lw=0.8,
            label="Condition A — No Schedule")
    ax5.bar(x,     rB["M"][sort_idx], w, color=COL["B"], alpha=0.85, edgecolor="white", lw=0.8,
            label="Condition B — Static Schedule")
    ax5.bar(x + w, rC["M"][sort_idx], w, color=COL["C"], alpha=0.90, edgecolor="white", lw=0.8,
            label="Condition C — Adaptive (Proposed)")
    ax5.set_xlabel("Topic (sorted by weightage, highest to lowest)")
    ax5.set_ylabel("Mastery Score (0\u20131)")
    ax5.set_xticks(x)
    ax5.set_xticklabels(labs, rotation=40, ha="right", fontsize=9)
    ax5.set_ylim(0, 1.05)
    ax5.legend(loc="upper right", fontsize=11)
    fig5.tight_layout(pad=0.6)
    fig5.savefig("pertopic_mastery.png", dpi=300, bbox_inches="tight")
    plt.close(fig5)
    print("Saved pertopic_mastery.png")

# ── Monte-Carlo across multiple seeds ────────────────────────────────────────
def run_monte_carlo(n_runs, base_seed):
    """
    Run the benchmark n_runs times with different seeds.
    Prints mean ± std for coverage and total hours per condition.
    Also saves a coverage distribution plot.
    """
    results = {"cA": [], "cB": [], "cC": [],
               "hA": [], "hB": [], "hC": []}

    for i in range(n_runs):
        np.random.seed(base_seed + i)
        comp = gen_compliance()
        rA = run_A(comp)
        rB = run_B(comp)
        rC = run_C(comp)
        results["cA"].append(coverage(rA["M"]))
        results["cB"].append(coverage(rB["M"]))
        results["cC"].append(coverage(rC["M"]))
        results["hA"].append(rA["tot"])
        results["hB"].append(rB["tot"])
        results["hC"].append(rC["tot"])
        if (i + 1) % max(1, n_runs // 10) == 0:
            print(f"  {i+1}/{n_runs} runs done")

    print(f"\n{'─'*62}")
    print(f"Monte-Carlo results over {n_runs} runs (seed range {base_seed}–{base_seed+n_runs-1})")
    print(f"{'─'*62}")
    for cond, ck, hk in [("A", "cA", "hA"), ("B", "cB", "hB"), ("C", "cC", "hC")]:
        cv = results[ck]
        hv = results[hk]
        print(f"Condition {cond}  coverage: {np.mean(cv):5.1f}% ± {np.std(cv):.1f}%  "
              f"| hours: {np.mean(hv):5.0f} ± {np.std(hv):.0f}")
    print(f"{'─'*62}")

    # Distribution plot
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 100, 21)
    ax.hist(results["cA"], bins=bins, alpha=0.6, color=COL["A"], label="A — No Schedule")
    ax.hist(results["cB"], bins=bins, alpha=0.6, color=COL["B"], label="B — Static")
    ax.hist(results["cC"], bins=bins, alpha=0.6, color=COL["C"], label="C — Adaptive")
    ax.set_xlabel("High-Priority Coverage (%) at Day 180")
    ax.set_ylabel("Count")
    ax.set_title(f"Coverage distribution over {n_runs} seeds")
    ax.legend()
    fig.tight_layout(pad=0.6)
    fig.savefig("coverage_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved coverage_distribution.png")

# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Stochastic discipline benchmark simulation")
    parser.add_argument("--seed",       type=int, default=42,
                        help="Random seed for single run (default: 42, matches paper)")
    parser.add_argument("--runs",       type=int, default=1,
                        help="Number of Monte-Carlo runs (default: 1)")
    parser.add_argument("--no-figures", action="store_true",
                        help="Skip saving figures (useful for batch runs)")
    args = parser.parse_args()

    if args.runs > 1:
        print(f"Running Monte-Carlo benchmark: {args.runs} seeds starting at {args.seed}")
        run_monte_carlo(args.runs, args.seed)
    else:
        np.random.seed(args.seed)
        comp = gen_compliance()
        print(f"Running single benchmark (seed={args.seed})...")
        rA = run_A(comp)
        rB = run_B(comp)
        rC = run_C(comp)

        cA = coverage(rA["M"])
        cB = coverage(rB["M"])
        cC = coverage(rC["M"])

        print(f"\nResults at day {DAYS}:")
        print(f"  Coverage:  A={cA:.1f}%  B={cB:.1f}%  C={cC:.1f}%")
        print(f"  Hours:     A={rA['tot']:.0f}  B={rB['tot']:.0f}  C={rC['tot']:.0f}")
        print(f"  Wtd mast:  A={rA['m'][-1]:.3f}  B={rB['m'][-1]:.3f}  C={rC['m'][-1]:.3f}")
        print(f"  Resets:    C={len(rC['resets'])} (days {rC['resets']})")

        if not args.no_figures:
            save_figures(rA, rB, rC)
            print("\nAll benchmark figures saved.")

if __name__ == "__main__":
    main()
