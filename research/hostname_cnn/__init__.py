"""Research harness for the hostname CNN (v4 programme).

Kept outside ``backend/`` on purpose: nothing here is imported by the app, and the deployed
v1-v3 code paths are unchanged. Run modules from the repository root, for example::

    python -m research.hostname_cnn.train --help

See ``docs/research/`` (added as phases complete) and ``cnn-research/phase1`` for the plan.
"""
