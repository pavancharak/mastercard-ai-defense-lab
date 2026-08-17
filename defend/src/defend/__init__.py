"""Defend pillar: trains and evaluates the fraud classifier consumed by the web prototype.

Reads generate/data/{transactions,disputes}.csv (read-only - this package
never writes into generate/), builds one unified transaction-level dataset
(see data.py for how the disputes table is joined in), trains a gradient-
boosted tree classifier, and evaluates it both in aggregate and broken out
per identify/attack-taxonomy.md category and vector.
"""
