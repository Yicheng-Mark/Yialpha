"""Multi-ticker batch concurrency for YiAgents.

This package is intentionally empty: importing ``yiagents.batch.locks`` (used by
core modules memory/stockstats) must NOT eagerly import ``yiagents.batch.runner``,
which would pull in the graph and create an import cycle. Import the runner
explicitly where needed::

    from yiagents.batch.runner import BatchRunner
"""
