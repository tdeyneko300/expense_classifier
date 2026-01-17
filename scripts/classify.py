from __future__ import annotations

import sys

from expense_classifier.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["classify", *sys.argv[1:]]))
