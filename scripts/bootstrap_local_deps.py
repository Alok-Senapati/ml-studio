#!/usr/bin/env python3
"""
Install local editable dependencies for the repository in a portable way.

Usage: python scripts/bootstrap_local_deps.py

This script installs any local packages located under the repository's
`libraries/` directory (for example `libraries/ml-core`) in editable mode
using the current Python interpreter. This avoids hard-coded absolute
file:// URLs in pyproject.toml and works across platforms.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_LIBS_DIR = REPO_ROOT / "libraries"

LOCAL_PACKAGES = [p for p in LOCAL_LIBS_DIR.iterdir() if p.is_dir()]

if not LOCAL_PACKAGES:
    print("No local libraries found under 'libraries/'. Nothing to install.")
    sys.exit(0)

for pkg in LOCAL_PACKAGES:
    print(f"Installing local package in editable mode: {pkg}")
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(pkg)], check=True)

print("All local editable packages installed.")
