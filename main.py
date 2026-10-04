import argparse
from scenarios.runner import run_monte_carlo, run_scenario

def main():
    parser = argparse.ArgumentParser(description="Run a room temperature scenario")
    parser.add_argument("--scenario", required=True, help="Path to YAML scenario file")
    parser.add_argument(
        "--monte-carlo",
        type=int,
        metavar="N",
        help="Run N seeded simulations and summarize their variability (N must be at least 2)",
    )
    args = parser.parse_args()
    if args.monte_carlo is None:
        run_scenario(args.scenario)
    elif args.monte_carlo < 2:
        parser.error("--monte-carlo must be at least 2")
    else:
        run_monte_carlo(args.scenario, args.monte_carlo)

if __name__ == "__main__":
    main()
