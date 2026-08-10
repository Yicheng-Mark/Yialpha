import contextlib
from importlib.metadata import PackageNotFoundError, version as _pkg_version

# Load .env files at package import so DEFAULT_CONFIG's env-var overlay
# (and every llm_clients consumer) sees the user's keys regardless of
# which entry point started the process. find_dotenv(usecwd=True) walks
# from the CWD, so the installed `yiagents` console script picks up
# the project's .env instead of stepping up from site-packages.
# load_dotenv defaults to override=False, so it never clobbers values
# the caller has already exported.
try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
    load_dotenv(find_dotenv(".env.enterprise", usecwd=True), override=False)
except ImportError:
    pass

# Expose the installed package version (declared in pyproject.toml). The fallback
# covers source checkouts / editable installs where the distribution metadata
# is absent, so `yiagents.__version__` never errors.
try:
    __version__ = _pkg_version("yiagents")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"
