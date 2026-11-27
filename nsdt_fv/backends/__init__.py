"""Checker backends. Everything outside backends/ talks to a checker only through
the CheckerBackend protocol of base.py (plan A.3)."""
from .base import CheckerBackend, Verdict, EvalResult, CATEGORIES  # noqa: F401
