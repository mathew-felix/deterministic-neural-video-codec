"""Backward-compatible import shim for benchmark_report_style.

The implementation was moved to tools/ as part of repository cleanup.
Some tests and external scripts still import `benchmark_report_style` from
the repository root, so this file re-exports the public symbols.
"""

from tools.benchmark_report_style import *  # noqa: F401,F403

