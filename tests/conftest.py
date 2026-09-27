import pytest

from edgeline.ev import payouts


@pytest.fixture(autouse=True)
def standard_prizepicks_format():
    """Tests price on the classic ladders; the Arena default is exercised by its own test."""
    old = payouts.PRIZEPICKS_FORMAT
    payouts.PRIZEPICKS_FORMAT = "standard"
    try:
        yield
    finally:
        payouts.PRIZEPICKS_FORMAT = old
