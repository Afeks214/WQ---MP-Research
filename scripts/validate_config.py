import sys
import yaml
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO))

from run_research import _load_config  # if this import fails, you must locate the real loader in run_research.py

def main():
    if len(sys.argv) < 2:
        print("FAIL: usage: python scripts/validate_config.py <config.yaml>")
        return 1
    p = Path(sys.argv[1]).expanduser().resolve()
    if not p.exists():
        print("FAIL: Config not found:", p)
        return 1

    try:
        cfg = _load_config(p)
        print("VALIDATION SUCCESS")
        return 0
    except Exception as e:
        print("VALIDATION ERROR:")
        print(e)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
