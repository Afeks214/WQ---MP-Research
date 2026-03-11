from __future__ import annotations

import argparse

from weightiz.cli import run_module5, run_module6, run_research, validate_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dispatch Weightiz package runners")
    parser.add_argument("job", choices=("research", "module5", "module6", "validate-artifacts"))
    parser.add_argument("job_args", nargs=argparse.REMAINDER, help="Arguments passed to the selected job")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.job == "research":
        run_research.main(args.job_args)
        return
    if args.job == "module5":
        run_module5.main(args.job_args)
        return
    if args.job == "module6":
        run_module6.main(args.job_args)
        return
    validate_artifacts.main(args.job_args)


if __name__ == "__main__":
    main()
