"""Multi-ticker batch concurrency for YiAlpha.

This package is intentionally empty: importing ``yialpha.batch.locks`` (used by
core modules memory/stockstats) must NOT eagerly import ``yialpha.batch.runner``,
which would pull in the graph and create an import cycle. Import the runner
explicitly where needed::

    from yialpha.batch.runner import BatchRunner
"""
