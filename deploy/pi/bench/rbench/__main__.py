"""`python -m rbench` entry point (needs deploy/pi/bench and deploy/pi on sys.path)."""

import sys

from rbench.cli import main

sys.exit(main())
