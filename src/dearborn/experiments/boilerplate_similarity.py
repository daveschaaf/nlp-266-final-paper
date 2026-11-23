"""Compatibility shim for :mod:`dearborn.labeling.similarity`."""

from dearborn.labeling.similarity import *  # noqa: F403
from dearborn.labeling.similarity import main


if __name__ == "__main__":
    raise SystemExit(main())
