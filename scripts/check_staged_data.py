"""LGPD guard: fail when corpus data files are staged for commit.

The pre-commit hook filters to ^data/ and passes staged filenames; any
filename reaching this script is a violation (docs/corpus/storage.md).
"""

import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
  """Return 1 when any staged path is corpus data; 0 otherwise."""
  paths = list(sys.argv[1:] if argv is None else argv)
  for path in paths:
    print(f"corpus data must never be committed (LGPD): {path}")
  return 1 if paths else 0


if __name__ == "__main__":
  raise SystemExit(main())
