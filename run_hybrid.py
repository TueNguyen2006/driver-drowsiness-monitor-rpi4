#!/usr/bin/env python3
"""Entry point for the Hybrid Driver Monitor System."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from hybrid_system.cli import main

if __name__ == "__main__":
    main()
