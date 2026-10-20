"""Smoke test confirming the package layout is importable.

Replace/extend once ingest, clean, and train have real logic to test.
"""


def test_healthy_package_imports():
    from healthy.data import clean, ingest
    from healthy.model import calibrate, train

    assert ingest is not None
    assert clean is not None
    assert train is not None
    assert calibrate is not None
