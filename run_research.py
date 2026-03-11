#!/usr/bin/env python3
from __future__ import annotations

import sys

from weightiz.cli import run_research as _impl


sys.modules[__name__] = _impl


if __name__ == "__main__":
    _impl.main()
