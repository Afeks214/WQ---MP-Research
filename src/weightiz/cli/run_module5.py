from __future__ import annotations

from weightiz.cli.run_research import build_parser, run_from_namespace


def main(argv: list[str] | None = None) -> None:
    parser = build_parser(description="Weightiz Module 5 runner")
    args = parser.parse_args(argv)
    run_from_namespace(args, enable_module6=False, mode="module5")


if __name__ == "__main__":
    main()
