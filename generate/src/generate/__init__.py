"""Generate pillar: builds the simulated dataset that mirrors identify/attack-taxonomy.md.

Every fraud vector (A1-A2, B1-B3, C1-C3, D1-D2, E1-E2) gets a dedicated generator
in vectors.py that samples the transaction-level signal described in the taxonomy
for that vector, plus a legitimate baseline population. dataset.py assembles the
labeled dataset consumed by the Defend pillar.
"""
