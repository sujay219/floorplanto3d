"""Shared pytest configuration."""

from pathlib import Path

import pytest

# Make the synthetic-plan fixtures importable as a package.
FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture(autouse=True)
def quiet_logging(caplog):
    """Keep pipeline logs out of test output while still allowing assertions."""
    caplog.set_level("INFO", logger="floorplanto3d")
