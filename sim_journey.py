"""
sim_journey.py  —  Single-student recovery journey (Figures 1 & 2)
===================================================================
Thin wrapper around sim_benchmark's consolidated simulator so both
journey and benchmark figures share identical parameters.

Usage
-----
  python sim_journey.py              # paper seed (7)
  python sim_journey.py --seed 42    # different seed
  python sim_journey.py --no-figures # print stats only
"""

import argparse

from sim_benchmark import run_journey


def main():
    parser = argparse.ArgumentParser(description="Single-student journey simulation")
    parser.add_argument("--seed",       type=int, default=7,
                        help="Random seed (default: 7, matches paper)")
    parser.add_argument("--no-figures", action="store_true",
                        help="Skip saving figures")
    args = parser.parse_args()

    run_journey(seed=args.seed, no_figures=args.no_figures)


if __name__ == "__main__":
    main()
