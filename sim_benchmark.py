"""
sim_benchmark.py  —  Five-condition ablation benchmark simulation
==================================================================
Reproduces Figures 3–5 and Table 1 from the paper.

Usage
-----
  python sim_benchmark.py                  # Monte-Carlo, 200 seeds (primary result)
  python sim_benchmark.py --seed 42        # single illustrative run only
  python sim_benchmark.py --runs 500       # Monte-Carlo over 500 seeds
  python sim_benchmark.py --runs 200 --seed 0   # reproducible Monte-Carlo
  python sim_benchmark.py --no-figures     # print tables only, skip plots

Conditions
----------
  A         No schedule   – random topic selection, 2 hrs/day, slow decay
  B         Static        – sequential topics, 4 hrs/day fixed, no priority
  C_hm      Hours-matched – C scheduling logic, flat 4 hrs/day (same as B),
                            revision enabled; isolates scheduling quality
                            + revision contribution from hours growth
  C_nr      No revision   – C scheduling logic, flat 4 hrs/day (same as B),
                            revision disabled; isolates pure scheduling quality
  C (full)  Adaptive      – proposed system: priority function, dependency
                            graph, geometric capacity recovery, psychological
                            reset, and revision

Monte Carlo is the primary result.  Single-seed runs are illustrative only
and are labelled as such in every figure title.

Ablation logic
--------------
  B  vs  C_nr   → pure scheduling quality   (same hours=4/day, no revision)
  C_nr vs C_hm  → revision contribution     (same hours=4/day, ± revision)
  C_hm vs C     → hours-growth contribution (flat 4/day vs geometric growth)
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
C_DONE_THRESH  = 0.55  # mastery at which topic is marked done
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
COL = {
    "A":    "#DC2626",   # red
    "B":    "#2563EB",   # blue
    "C_hm": "#16A34A",   # medium green  (hours-matched)
    "C_nr": "#15803D",   # dark green    (no revision)
    "C":    "#166534",   # deepest green (full adaptive)
}

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

    sessions = sum(1 for h in hlog if h > 0)
    return {"M": M, "m": mlog, "tot": sum(hlog), "sessions": sessions}

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

    sessions = sum(1 for h in hlog if h > 0)
    return {"M": M, "m": mlog, "tot": sum(hlog), "sessions": sessions}

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
            D = np.clip(h / K_theoretical, 0, 1) if K_theoretical > 0 else 0.0
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

    return {"M": M, "m": mlog, "resets": resets, "tot": sum(hlog),
            "sessions": sum(1 for h in hlog if h > 0)}

# ── Condition C_hours_matched — C scheduling, flat 4 hrs/day (same as B) ──────
def run_C_hours_matched(comp):
    """
    Condition C with full C scheduling logic and flat 4 hrs/day (same as B).
    Revision enabled.  Uses B_HOURS_PER_DAY per compliant day so that both
    per-day hours and total hours match Condition B exactly, eliminating the
    temporal-distribution confound of the old budget-cap approach.

    C_nr vs C_hm isolates the revision contribution alone (same hours and
    scheduling algorithm, ± revision).
    C_hm vs C isolates the hours-growth contribution (flat 4/day vs geometric).
    """
    M    = np.full(N, 0.2)
    n    = np.zeros(N, int)
    done = np.zeros(N, bool)

    K      = C_K0
    r      = C_R_BASE
    r_prev = C_R_BASE

    consec_missed = 0
    streak_buf    = []
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

        studied_topics = []

        if c > 0:
            h = c * B_HOURS_PER_DAY
            consec_missed = 0

            # ── Priority scores ──────────────────────────────────────────────
            P = W_arr * D_arr * (1.0 - M)
            for i in range(N):
                if M[i] >= C_DONE_THRESH:
                    done[i] = True
                if M[i] < C_UNDONE_THRESH and done[i]:
                    done[i] = False
                if done[i]:
                    P[i] = 0.0

            P2 = P.copy()
            for j in range(N):
                for p in TOPICS[j]["prereqs"]:
                    P2[p] += C_PROP_GAMMA * C_PROP_LAMBDA * P[j]
            P = P2

            eligible = [
                i for i in range(N)
                if all(M[p] >= C_PREREQ_MIN for p in TOPICS[i]["prereqs"])
                and not done[i]
            ]
            if not eligible:
                eligible = [i for i in range(N) if not done[i]]
            if not eligible:
                eligible = list(range(N))

            neglected = [i for i in range(N) if M[i] <= C_NEGLECT_MARK + 1e-6]
            if neglected and h > C_NEGLECT_HRS:
                eligible = list(set(eligible + neglected[:3]))

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

            # Periodic revision (enabled; C_nr vs C_hm isolates this block)
            done_list = [i for i in range(N) if done[i]]
            if day % C_REVISION_EVERY == 0 and done_list and h > 0:
                for rev in sorted(done_list, key=lambda i: M[i])[:C_REVISION_N]:
                    s = np.random.uniform(C_REV_SCORE_LO, C_REV_SCORE_HI)
                    n[rev] += 1
                    alpha   = 1.0 / n[rev]
                    M[rev]  = np.clip(M[rev] + alpha * (s - M[rev]), 0, 1)
                    studied_topics.append(rev)

            # D = c since h = c * B_HOURS_PER_DAY
            D = c
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

        # Knowledge decay for unstudied topics
        for j in range(N):
            if j not in studied_topics:
                M[j] *= C_DECAY
        M = np.clip(M, 0, 1)

        K = 0.8 * K + 0.2 * h
        hlog.append(h)
        mlog.append(weighted_mastery(M))

    return {"M": M, "m": mlog, "resets": resets, "tot": sum(hlog),
            "sessions": sum(1 for h in hlog if h > 0)}


# ── Condition C_no_revision — C scheduling, flat 4 hrs/day, revision off ──────
def run_C_no_revision(comp):
    """
    Condition C with full C scheduling logic, flat 4 hrs/day (same as B),
    and revision disabled.  Isolates the pure scheduling-quality contribution
    over B (same hours=4/day, no revision in either condition).

    C_nr vs B   → pure scheduling quality  (same hours=4/day, no revision)
    C_nr vs C_hm → revision contribution   (same hours=4/day, ± revision)
    """
    M    = np.full(N, 0.2)
    n    = np.zeros(N, int)
    done = np.zeros(N, bool)

    K      = C_K0
    r      = C_R_BASE
    r_prev = C_R_BASE

    consec_missed = 0
    streak_buf    = []
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

        studied_topics = []

        if c > 0:
            h = c * B_HOURS_PER_DAY
            consec_missed = 0

            P = W_arr * D_arr * (1.0 - M)
            for i in range(N):
                if M[i] >= C_DONE_THRESH:
                    done[i] = True
                if M[i] < C_UNDONE_THRESH and done[i]:
                    done[i] = False
                if done[i]:
                    P[i] = 0.0

            P2 = P.copy()
            for j in range(N):
                for p in TOPICS[j]["prereqs"]:
                    P2[p] += C_PROP_GAMMA * C_PROP_LAMBDA * P[j]
            P = P2

            eligible = [
                i for i in range(N)
                if all(M[p] >= C_PREREQ_MIN for p in TOPICS[i]["prereqs"])
                and not done[i]
            ]
            if not eligible:
                eligible = [i for i in range(N) if not done[i]]
            if not eligible:
                eligible = list(range(N))

            neglected = [i for i in range(N) if M[i] <= C_NEGLECT_MARK + 1e-6]
            if neglected and h > C_NEGLECT_HRS:
                eligible = list(set(eligible + neglected[:3]))

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

            # Revision block intentionally omitted

            # D = c since h = c * B_HOURS_PER_DAY
            D = c
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

        for j in range(N):
            if j not in studied_topics:
                M[j] *= C_DECAY
        M = np.clip(M, 0, 1)

        K = 0.8 * K + 0.2 * h
        hlog.append(h)
        mlog.append(weighted_mastery(M))

    return {"M": M, "m": mlog, "resets": resets, "tot": sum(hlog),
            "sessions": sum(1 for h in hlog if h > 0)}


# ── Ablation table printer ────────────────────────────────────────────────────
def print_ablation_table(rA, rB, rC_hm, rC_nr, rC, label=""):
    """
    Print a formatted ablation table for a single seed run.
    Columns: Condition | Coverage % | Hours | Sessions | Wtd-Mastery
    """
    rows = [
        ("A   — No schedule",      rA),
        ("B   — Static",           rB),
        ("C_hm — Hours-matched",   rC_hm),
        ("C_nr — No revision",     rC_nr),
        ("C   — Full adaptive",    rC),
    ]
    sep = "─" * 72
    hdr = f"{'Condition':<28} {'Coverage':>10} {'Hours':>8} {'Sessions':>10} {'Wtd-Mast':>10}"
    if label:
        print(f"\n{label}")
    print(f"\n{sep}")
    print(hdr)
    print(sep)
    for lbl, r in rows:
        cov = coverage(r["M"])
        hrs = r["tot"]
        ses = r["sessions"]
        wm  = r["m"][-1]
        print(f"{lbl:<28} {cov:>9.1f}% {hrs:>8.0f} {ses:>10} {wm:>10.3f}")
    print(sep)
    # Derived interpretation lines
    b_cov  = coverage(rB["M"])
    hm_cov = coverage(rC_hm["M"])
    nr_cov = coverage(rC_nr["M"])
    c_cov  = coverage(rC["M"])
    print(f"  Pure scheduling gain  (C_nr − B):       {nr_cov - b_cov:+.1f}%")
    print(f"  Revision contribution (C_hm − C_nr):    {hm_cov - nr_cov:+.1f}%")
    print(f"  Hours-growth gain     (C_full − C_hm):  {c_cov  - hm_cov:+.1f}%")
    print(sep)

# ── Single-run figures ────────────────────────────────────────────────────────
def save_figures(rA, rB, rC_hm, rC_nr, rC, seed):
    """Save illustrative single-seed figures (Figures 3–5)."""
    illu = f"Illustrative — seed {seed}"
    dx   = np.arange(DAYS)

    # Fig 3 — weighted mastery over time (14-day rolling average)
    fig3, ax3 = plt.subplots(figsize=(8, 3.5))
    ax3.plot(dx, rolling14(rA["m"]),    color=COL["A"],    ls="--", lw=2.0, alpha=0.9,
             label="A — No Schedule")
    ax3.plot(dx, rolling14(rB["m"]),    color=COL["B"],    ls=":",  lw=2.0, alpha=0.9,
             label="B — Static (4 hrs/day)")
    ax3.plot(dx, rolling14(rC_hm["m"]), color=COL["C_hm"], ls="-.", lw=1.8, alpha=0.85,
             label="C_hm — Hours-matched")
    ax3.plot(dx, rolling14(rC_nr["m"]), color=COL["C_nr"], ls="-.", lw=1.8, alpha=0.85,
             label="C_nr — No revision")
    ax3.plot(dx, rolling14(rC["m"]),    color=COL["C"],    ls="-",  lw=2.4, alpha=0.95,
             label="C — Full Adaptive")
    for rd in rC["resets"]:
        ax3.axvline(rd, color=COL["C"], ls=":", lw=0.9, alpha=0.45)
        ax3.annotate("\u21ba", xy=(rd + 1, 0.03), fontsize=10, color=COL["C"], alpha=0.8)
    ax3.set_xlabel("Day")
    ax3.set_ylabel("Weighted Mastery Score (0\u20131)")
    ax3.set_title(f"Weighted mastery over time  ({illu})")
    ax3.set_xlim(0, DAYS)
    ax3.set_ylim(0, 1.0)
    ax3.legend(loc="lower right", fontsize=10, framealpha=0.9)
    fig3.tight_layout(pad=0.6)
    fig3.savefig("weighted_mastery.png", dpi=300, bbox_inches="tight")
    plt.close(fig3)
    print("Saved weighted_mastery.png")

    cA   = coverage(rA["M"])
    cB   = coverage(rB["M"])
    cC_hm = coverage(rC_hm["M"])
    cC_nr = coverage(rC_nr["M"])
    cC   = coverage(rC["M"])

    # Fig 4 — coverage bar chart (all 5 conditions)
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    labels = [
        "A\n(No Schedule)", "B\n(Static, 4 hrs)",
        "C_hm\n(Hrs-matched)", "C_nr\n(No revision)", "C\n(Full Adaptive)",
    ]
    vals   = [cA, cB, cC_hm, cC_nr, cC]
    colors = [COL["A"], COL["B"], COL["C_hm"], COL["C_nr"], COL["C"]]
    bars = ax4.bar(labels, vals, color=colors, edgecolor="white",
                   linewidth=1.5, width=0.55, alpha=0.92)
    for bar, val in zip(bars, vals):
        ax4.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            f"{val:.0f}%", ha="center", va="bottom",
            fontsize=12, fontweight="bold", color="#1F2937",
        )
    ax4.set_ylabel("High-Priority Coverage (%)")
    ax4.set_title(f"High-priority coverage — ablation  ({illu})")
    ax4.set_ylim(0, 115)
    fig4.tight_layout(pad=0.6)
    fig4.savefig("coverage_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig4)
    print("Saved coverage_bar.png")

    # Fig 5 — per-topic mastery (A / B / C full only, to keep the chart readable)
    sort_idx = np.argsort(W_arr)[::-1]
    labs     = [TOPICS[i]["name"] for i in sort_idx]
    fig5, ax5 = plt.subplots(figsize=(14, 4))
    x = np.arange(N)
    w = 0.28
    ax5.bar(x - w, rA["M"][sort_idx], w, color=COL["A"],  alpha=0.85, edgecolor="white", lw=0.8,
            label="A — No Schedule")
    ax5.bar(x,     rB["M"][sort_idx], w, color=COL["B"],  alpha=0.85, edgecolor="white", lw=0.8,
            label="B — Static Schedule")
    ax5.bar(x + w, rC["M"][sort_idx], w, color=COL["C"],  alpha=0.90, edgecolor="white", lw=0.8,
            label="C — Full Adaptive")
    ax5.set_xlabel("Topic (sorted by weightage, highest to lowest)")
    ax5.set_ylabel("Mastery Score (0\u20131)")
    ax5.set_title(f"Per-topic mastery  ({illu})")
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
    Run the five-condition ablation benchmark n_runs times.

    Primary outputs
    ---------------
    • Printed ablation table: mean ± std for coverage and hours per condition.
    • ablation_table.png — bar chart with error bars (the paper's primary figure).
    • coverage_distribution.png — stacked histogram of coverage distributions.

    Interpretation guide printed below the table:
      B  vs  C_hm   → pure scheduling quality  (same hours, different algorithm)
      C_hm vs C_nr  → revision contribution    (same hours + scheduling, ± revision)
      C_nr vs C     → hours-growth advantage   (revision off vs full system)
    """
    results = {
        "cA": [],    "cB": [],    "cC_hm": [],  "cC_nr": [],  "cC": [],
        "hA": [],    "hB": [],    "hC_hm": [],  "hC_nr": [],  "hC": [],
        "sA": [],    "sB": [],    "sC_hm": [],  "sC_nr": [],  "sC": [],
    }

    for i in range(n_runs):
        np.random.seed(base_seed + i)
        comp  = gen_compliance()
        rA    = run_A(comp)
        rB    = run_B(comp)
        rC_hm = run_C_hours_matched(comp)
        rC_nr = run_C_no_revision(comp)
        rC    = run_C(comp)

        results["cA"].append(coverage(rA["M"]))
        results["cB"].append(coverage(rB["M"]))
        results["cC_hm"].append(coverage(rC_hm["M"]))
        results["cC_nr"].append(coverage(rC_nr["M"]))
        results["cC"].append(coverage(rC["M"]))

        results["hA"].append(rA["tot"])
        results["hB"].append(rB["tot"])
        results["hC_hm"].append(rC_hm["tot"])
        results["hC_nr"].append(rC_nr["tot"])
        results["hC"].append(rC["tot"])

        results["sA"].append(rA["sessions"])
        results["sB"].append(rB["sessions"])
        results["sC_hm"].append(rC_hm["sessions"])
        results["sC_nr"].append(rC_nr["sessions"])
        results["sC"].append(rC["sessions"])

        if (i + 1) % max(1, n_runs // 10) == 0:
            print(f"  {i+1}/{n_runs} runs done")

    sep = "─" * 80
    print(f"\n{sep}")
    print(f"Monte-Carlo ablation  |  {n_runs} seeds  (range {base_seed}–{base_seed+n_runs-1})")
    print(sep)
    print(f"{'Condition':<26} {'Coverage (%)':>14} {'Hours':>12} {'Sessions':>10}")
    print(sep)
    cond_info = [
        ("A   — No schedule",    "cA",    "hA",    "sA"),
        ("B   — Static",         "cB",    "hB",    "sB"),
        ("C_hm — Hrs-matched",   "cC_hm", "hC_hm", "sC_hm"),
        ("C_nr — No revision",   "cC_nr", "hC_nr", "sC_nr"),
        ("C   — Full adaptive",  "cC",    "hC",    "sC"),
    ]
    for lbl, ck, hk, sk in cond_info:
        cv = results[ck]
        hv = results[hk]
        sv = results[sk]
        print(f"{lbl:<26} {np.mean(cv):6.1f}% ±{np.std(cv):4.1f}%  "
              f"{np.mean(hv):6.0f} ±{np.std(hv):3.0f}  "
              f"{np.mean(sv):7.0f} ±{np.std(sv):.0f}")
    print(sep)
    # Interpretation
    bm   = np.mean(results["cB"])
    hmm  = np.mean(results["cC_hm"])
    nrm  = np.mean(results["cC_nr"])
    cm   = np.mean(results["cC"])
    print(f"  Pure scheduling gain  (C_nr − B):       {nrm - bm:+.1f}%  (same hours=4/day, no revision)")
    print(f"  Revision contribution (C_hm − C_nr):    {hmm - nrm:+.1f}%  (same hours=4/day, ± revision)")
    print(f"  Hours-growth gain     (C_full − C_hm):  {cm  - hmm:+.1f}%  (flat 4/day vs geometric growth)")
    print(sep)

    # ── Ablation bar chart (primary figure) ───────────────────────────────────
    fig_abl, ax_abl = plt.subplots(figsize=(10, 5))
    cond_labels = [
        "A\n(No sched.)",
        "B\n(Static)",
        "C_hm\n(Hrs-matched)",
        "C_nr\n(No revision)",
        "C\n(Full)",
    ]
    means  = [np.mean(results[k]) for k in ["cA", "cB", "cC_hm", "cC_nr", "cC"]]
    stds   = [np.std(results[k])  for k in ["cA", "cB", "cC_hm", "cC_nr", "cC"]]
    colors = [COL["A"], COL["B"], COL["C_hm"], COL["C_nr"], COL["C"]]
    bars = ax_abl.bar(
        cond_labels, means, yerr=stds, capsize=6,
        color=colors, alpha=0.88, width=0.55,
        error_kw={"linewidth": 1.8, "ecolor": "#374151"},
    )
    for bar, m, s in zip(bars, means, stds):
        ax_abl.text(
            bar.get_x() + bar.get_width() / 2,
            m + s + 1.5,
            f"{m:.1f}%",
            ha="center", va="bottom", fontsize=11, fontweight="bold", color="#1F2937",
        )
    ax_abl.set_ylabel("High-Priority Coverage (%) — mean ± std")
    ax_abl.set_title(
        f"Ablation over {n_runs} seeds  "
        f"(seed range {base_seed}–{base_seed+n_runs-1})  —  PRIMARY RESULT",
        fontsize=12,
    )
    ax_abl.set_ylim(0, 115)
    fig_abl.tight_layout(pad=0.6)
    fig_abl.savefig("ablation_table.png", dpi=300, bbox_inches="tight")
    plt.close(fig_abl)
    print("Saved ablation_table.png")

    # ── Coverage distribution histogram ───────────────────────────────────────
    fig_dist, ax_dist = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 100, 21)
    ax_dist.hist(results["cA"],    bins=bins, alpha=0.55, color=COL["A"],    label="A — No Schedule")
    ax_dist.hist(results["cB"],    bins=bins, alpha=0.55, color=COL["B"],    label="B — Static")
    ax_dist.hist(results["cC_hm"], bins=bins, alpha=0.55, color=COL["C_hm"], label="C_hm — Hrs-matched")
    ax_dist.hist(results["cC_nr"], bins=bins, alpha=0.55, color=COL["C_nr"], label="C_nr — No revision")
    ax_dist.hist(results["cC"],    bins=bins, alpha=0.55, color=COL["C"],    label="C — Full Adaptive")
    ax_dist.set_xlabel("High-Priority Coverage (%) at Day 180")
    ax_dist.set_ylabel("Count")
    ax_dist.set_title(f"Coverage distribution over {n_runs} seeds")
    ax_dist.legend(fontsize=10)
    fig_dist.tight_layout(pad=0.6)
    fig_dist.savefig("coverage_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig_dist)
    print("Saved coverage_distribution.png")

# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Stochastic discipline benchmark simulation")
    parser.add_argument("--seed",       type=int, default=42,
                        help="Base random seed (default: 42, matches paper)")
    parser.add_argument("--runs",       type=int, default=200,
                        help="Number of Monte-Carlo runs (default: 200).  "
                             "Set to 1 for a single illustrative run.")
    parser.add_argument("--no-figures", action="store_true",
                        help="Skip saving figures (useful for batch runs)")
    args = parser.parse_args()

    # ── Monte-Carlo (primary result) ─────────────────────────────────────────
    if args.runs > 1:
        print(f"Running Monte-Carlo ablation: {args.runs} seeds "
              f"(base seed {args.seed})")
        run_monte_carlo(args.runs, args.seed)

    # ── Single illustrative seed ──────────────────────────────────────────────
    np.random.seed(args.seed)
    comp  = gen_compliance()
    label = (f"Single illustrative run  (seed={args.seed})"
             if args.runs > 1
             else f"Single run  (seed={args.seed})")
    print(f"\n{label}")
    print(f"Forced absence windows: {ABSENCE_WINDOWS}")

    rA    = run_A(comp)
    rB    = run_B(comp)
    rC_hm = run_C_hours_matched(comp)
    rC_nr = run_C_no_revision(comp)
    rC    = run_C(comp)

    print_ablation_table(rA, rB, rC_hm, rC_nr, rC, label=label)
    print(f"  C resets: {len(rC['resets'])} (days {rC['resets']})")

    if not args.no_figures:
        save_figures(rA, rB, rC_hm, rC_nr, rC, seed=args.seed)
        print("\nAll benchmark figures saved.")


if __name__ == "__main__":
    main()
