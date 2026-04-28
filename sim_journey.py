"""
sim_journey.py  —  Single-student recovery journey (Figures 1 & 2)
====================================================================
Simulates one student following the proposed adaptive system over 180 days,
showing capacity recovery and discipline score across two forced absence
windows.

Usage
-----
  python sim_journey.py              # paper seed (7)
  python sim_journey.py --seed 42    # different seed
  python sim_journey.py --no-figures # print stats only
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
    {"name": "Properties of Matter", "W": 0.60, "D": 0.55, "prereqs": []},
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
P_FULL    = 0.55
P_PARTIAL = 0.20   # (else missed)
PARTIAL_LO, PARTIAL_HI = 0.2, 0.7

# Forced absence windows (0-indexed days, inclusive)
ABSENCE_WINDOWS = [(20, 22), (85, 88)]

# Capacity recovery
K0       = 2.0    # initial capacity (hrs/day)
K_TARGET = 7.0    # target capacity
R_BASE   = 0.05   # base daily growth rate

# Mastery / scheduling thresholds
DONE_THRESH   = 0.82   # mastery at which topic is marked done
UNDONE_THRESH = 0.62   # mastery below which a done topic is re-opened
PREREQ_MIN    = 0.28   # minimum prereq mastery to unlock a topic
DECAY         = 0.995  # daily knowledge decay for unstudied topics

# Score draws per session (improve with repetition)
SCORE_EARLY  = (0.38, 0.58)   # first 1–3 sessions
SCORE_MID    = (0.48, 0.75)   # sessions 4–8
SCORE_LATE   = (0.58, 0.95)   # sessions 9+

# Priority propagation
PROP_GAMMA  = 0.9
PROP_LAMBDA = 0.7

# Session allocation
SESSION_LEN   = 1.5    # hrs per slot
MIN_SLOT_HRS  = 0.18   # discard slots shorter than this

# Revision of completed topics
REVISION_EVERY = 5     # every N days
REVISION_N     = 2     # how many topics to review
REV_SCORE = (0.75, 0.95)

# Growth rate adaptation
RESET_DAYS  = 3        # consecutive missed days → psychological reset
STREAK_DAYS = 7        # streak before r boost
R_BOOST     = 1.1

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

# ── Compliance generator ──────────────────────────────────────────────────────
def gen_compliance():
    """
    Generate DAYS-length compliance sequence.
    Forced absence windows (ABSENCE_WINDOWS) are set directly without
    consuming random numbers, matching the original random-state behaviour.
    """
    forced_absent = set()
    for lo, hi in ABSENCE_WINDOWS:
        for d in range(lo, hi + 1):
            forced_absent.add(d)

    seq = []
    for day in range(DAYS):
        if day in forced_absent:
            seq.append(0.0)
        else:
            r = np.random.random()
            if r < P_FULL:
                seq.append(1.0)
            elif r < P_FULL + P_PARTIAL:
                seq.append(float(np.random.uniform(PARTIAL_LO, PARTIAL_HI)))
            else:
                seq.append(0.0)
    return seq

# ── Journey simulation ────────────────────────────────────────────────────────
def run_journey(comp):
    """
    Simulate the proposed adaptive system for one student over DAYS days.

    Returns
    -------
    dict with keys:
      K_smooth   – daily smoothed capacity (EWMA of actual hours)
      K_theory   – theoretical K(t) curve
      D_daily    – daily discipline score
      H_daily    – actual hours studied per day
      resets     – list of days on which a psychological reset fired
    """
    M    = np.full(N, 0.2)
    n    = np.zeros(N, int)
    done = np.zeros(N, bool)

    K       = K0
    r       = R_BASE
    r_prev  = R_BASE

    consec_missed = 0
    streak_buf    = []
    resets        = []

    K_smooth  = []
    K_theory  = []
    D_daily   = []
    H_daily   = []

    for day in range(DAYS):
        c = comp[day]

        # ── Psychological reset ──────────────────────────────────────────────
        if consec_missed >= RESET_DAYS:
            resets.append(day)
            consec_missed = 0
            r = R_BASE

        # Effective growth rate: at least R_BASE * 0.3 even after penalty
        r_eff     = max(r, R_BASE * 0.3)
        K_th      = min(K0 * (1 + R_BASE) ** day, K_TARGET)   # fixed reference curve
        K_th_eff  = min(K0 * (1 + r_eff)  ** day, K_TARGET)   # actual scheduled hrs
        K_theory.append(K_th)

        studied_topics = []

        if c > 0:
            h = c * K_th_eff
            consec_missed = 0

            # ── Priority scores ──────────────────────────────────────────────
            P = W_arr * D_arr * (1.0 - M)
            for i in range(N):
                if M[i] >= DONE_THRESH:
                    done[i] = True
                if M[i] < UNDONE_THRESH and done[i]:
                    done[i] = False
                if done[i]:
                    P[i] = 0.0

            # Priority propagation
            P2 = P.copy()
            for j in range(N):
                for p in TOPICS[j]["prereqs"]:
                    P2[p] += PROP_GAMMA * PROP_LAMBDA * P[j]
            P = P2

            # Eligible topics
            eligible = [
                i for i in range(N)
                if all(M[p] >= PREREQ_MIN for p in TOPICS[i]["prereqs"])
                and not done[i]
            ]
            if not eligible:
                eligible = [i for i in range(N) if not done[i]]
            if not eligible:
                eligible = list(range(N))

            # Allocate hours across top-priority topics
            top_topics = sorted(eligible, key=lambda i: P[i], reverse=True)
            n_slots    = min(len(top_topics), max(2, int(h / SESSION_LEN)))
            top        = top_topics[:n_slots]
            pv         = np.array([P[i] + 1e-9 for i in top])
            pv        /= pv.sum()
            hours_each = pv * h

            for idx, t in enumerate(top):
                if hours_each[idx] < MIN_SLOT_HRS:
                    continue
                # Score improves with repetition
                if n[t] <= 3:
                    s = np.random.uniform(*SCORE_EARLY)
                elif n[t] <= 8:
                    s = np.random.uniform(*SCORE_MID)
                else:
                    s = np.random.uniform(*SCORE_LATE)
                n[t] += 1
                alpha  = 1.0 / n[t]
                M[t]   = np.clip(M[t] + alpha * (s - M[t]), 0, 1)
                studied_topics.append(t)

            # Periodic revision
            done_list = [i for i in range(N) if done[i]]
            if day % REVISION_EVERY == 0 and done_list and h > 0:
                for rev in sorted(done_list, key=lambda i: M[i])[:REVISION_N]:
                    s = np.random.uniform(*REV_SCORE)
                    n[rev] += 1
                    alpha   = 1.0 / n[rev]
                    M[rev]  = np.clip(M[rev] + alpha * (s - M[rev]), 0, 1)
                    studied_topics.append(rev)

            # ── Update growth rate ───────────────────────────────────────────
            D = np.clip(h / K_th, 0, 1) if K_th > 0 else 0.0
            if D >= 0.8:
                r = max(r_prev, R_BASE)
            elif D > 0:
                r_prev = r
                r = max(r / 2, R_BASE * 0.3)
            streak_buf.append(D)
            if len(streak_buf) > STREAK_DAYS:
                streak_buf.pop(0)
            if len(streak_buf) == STREAK_DAYS and all(d >= 0.8 for d in streak_buf):
                r *= R_BOOST

        else:
            h = 0.0
            D = 0.0
            consec_missed += 1
            r_prev = r
            r = R_BASE * 0.1
            streak_buf.append(0.0)
            if len(streak_buf) > STREAK_DAYS:
                streak_buf.pop(0)

        # Knowledge decay
        for j in range(N):
            if j not in studied_topics:
                M[j] *= DECAY
        M = np.clip(M, 0, 1)

        K = 0.8 * K + 0.2 * h
        K_smooth.append(float(K))
        H_daily.append(float(h))
        D_daily.append(float(D))

    return {
        "K_smooth": K_smooth,
        "K_theory": K_theory,
        "D_daily":  D_daily,
        "H_daily":  H_daily,
        "resets":   resets,
    }

# ── Figures ───────────────────────────────────────────────────────────────────
def save_figures(result):
    dx  = np.arange(DAYS)
    s7  = lambda x: pd.Series(x).rolling(7, min_periods=1).mean().tolist()

    # Figure 1 — capacity recovery
    fig1, ax1 = plt.subplots(figsize=(8, 3.5))
    ax1.fill_between(dx, s7(result["H_daily"]), alpha=0.18, color="#16A34A")
    ax1.plot(dx, s7(result["H_daily"]), color="#16A34A", lw=2.2,
             label="Actual hours studied (7-day avg)")
    # Plot theoretical curve only up to where it hits K_TARGET
    plateau_day = next((i for i, k in enumerate(result["K_theory"]) if k >= K_TARGET - 0.01), DAYS)
    ax1.plot(dx[:plateau_day], result["K_theory"][:plateau_day],
             color="black", lw=1.5, ls="--", alpha=0.5, label="K(t) theoretical")
    ax1.axhline(K0, color="#6B7280", lw=1.2, ls=":", alpha=0.6,
                label=f"K\u2080 = {K0} hrs (baseline)")
    for rd in result["resets"]:
        ax1.axvline(rd, color="#DC2626", lw=1.4, ls=":", alpha=0.7)
        ax1.annotate("Reset", xy=(rd + 1.5, 0.35), fontsize=10,
                     color="#DC2626", fontstyle="italic")
    ax1.set_xlabel("Day")
    ax1.set_ylabel("Hours per Day")
    ax1.set_xlim(0, DAYS)
    ax1.set_ylim(0, 8.5)
    ax1.legend(loc="upper left", fontsize=11, framealpha=0.9)
    fig1.tight_layout(pad=0.6)
    fig1.savefig("capacity_recovery.png", dpi=300, bbox_inches="tight")
    plt.close(fig1)
    print("Saved capacity_recovery.png")

    # Figure 2 — discipline score
    fig2, ax2 = plt.subplots(figsize=(8, 3.2))
    ax2.fill_between(dx, s7(result["D_daily"]), alpha=0.15, color="#2563EB")
    ax2.plot(dx, s7(result["D_daily"]), color="#2563EB", lw=2.2,
             label="D = Actual / Scheduled hours (7-day avg)")
    for rd in result["resets"]:
        ax2.axvline(rd, color="#DC2626", lw=1.4, ls=":", alpha=0.7)
        ax2.annotate("Reset", xy=(rd + 1.5, 0.06), fontsize=10,
                     color="#DC2626", fontstyle="italic")
    ax2.set_xlabel("Day")
    ax2.set_ylabel("Discipline Score (0\u20131)")
    ax2.set_xlim(0, DAYS)
    ax2.set_ylim(0, 1.1)
    ax2.legend(loc="lower right", fontsize=11, framealpha=0.9)
    fig2.tight_layout(pad=0.6)
    fig2.savefig("discipline_journey.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print("Saved discipline_journey.png")

# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Single-student journey simulation")
    parser.add_argument("--seed",       type=int, default=7,
                        help="Random seed (default: 7, matches paper)")
    parser.add_argument("--no-figures", action="store_true",
                        help="Skip saving figures")
    args = parser.parse_args()

    np.random.seed(args.seed)
    comp   = gen_compliance()
    result = run_journey(comp)

    print(f"Seed: {args.seed}")
    print(f"Resets fired: {result['resets']}")
    print(f"Final smoothed capacity: {result['K_smooth'][-1]:.2f} hrs/day")

    if not args.no_figures:
        save_figures(result)
        print("\nAll journey figures saved.")

if __name__ == "__main__":
    main()
