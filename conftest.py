"""
Root-level conftest.py — ensures the project root is on sys.path so that
`from src.X import Y` works in all test files regardless of how pytest is
invoked.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
