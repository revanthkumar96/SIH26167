"""HTTP application: uploads, queries, trace streaming and benchmark runs."""

from satquery.api.settings import Settings

__all__ = ["Settings", "create_app"]


def create_app(settings: Settings | None = None):
    """Lazy re-export so importing the package does not pull in FastAPI."""
    from satquery.api.app import create_app as _create_app

    return _create_app(settings)
