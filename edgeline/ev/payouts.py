"""Payout ladders for fixed-payout pick'em books.

Each ladder is indexed by the number of legs that hit and gives the NET profit
per 1 unit staked (so -1 is a total loss, 9 means the slip returned 10x).

The PrizePicks and Underdog ladders below are the exact arrays shipped in
LCSLarry's public EV calculator (ev.lcslarry.com) as of September 2026.
Books change these from time to time; verify against the app before relying
on a ladder for real money.
"""
from __future__ import annotations

from dataclasses import dataclass

PRIZEPICKS = {
    "POWER": {
        2: [-1, -1, 2],
        3: [-1, -1, -1, 5],
        4: [-1, -1, -1, -1, 9],
        5: [-1, -1, -1, -1, -1, 19],
        6: [-1, -1, -1, -1, -1, -1, 36.5],
    },
    "FLEX": {
        3: [-1, -1, 0.25, 1.25],
        4: [-1, -1, -1, 0.5, 5],
        5: [-1, -1, -1, -0.6, 1, 9],
        6: [-1, -1, -1, -1, -0.6, 1, 24],
    },
}

UNDERDOG = {
    "POWER": {
        2: [-1, -1, 2.5],
        3: [-1, -1, -1, 5.5],
        4: [-1, -1, -1, -1, 9],
        5: [-1, -1, -1, -1, -1, 19],
        6: [-1, -1, -1, -1, -1, -1, 34],
        7: [-1, -1, -1, -1, -1, -1, -1, 64],
        8: [-1, -1, -1, -1, -1, -1, -1, -1, 119],
    },
    "FLEX": {
        3: [-1, -1, 0, 2],
        4: [-1, -1, -1, 0.5, 5],
        5: [-1, -1, -1, -1, 1.5, 9],
        6: [-1, -1, -1, -1, -0.75, 1.6, 24],
        7: [-1, -1, -1, -1, -1, -0.5, 1.75, 39],
        8: [-1, -1, -1, -1, -1, -1, 0, 2, 79],
    },
}

LADDERS = {"prizepicks": PRIZEPICKS, "underdog": UNDERDOG}

# Per-leg implied decimal odds used to price a single leg on a fixed-payout book.
# LCSLarry's convention: derive from the 4-pick POWER payout, i.e. payout ** (1/4).
REFERENCE_PARLAY_SIZE = 4


@dataclass(frozen=True)
class LegPrice:
    book: str
    decimal_odds: float

    @property
    def implied_prob(self) -> float:
        return 1.0 / self.decimal_odds


def ladder(book: str, slip_type: str, size: int) -> list[float]:
    try:
        return list(LADDERS[book.lower()][slip_type.upper()][int(size)])
    except KeyError as exc:
        raise KeyError(f"no ladder for book={book} type={slip_type} size={size}") from exc


def leg_decimal_odds(book: str, reference_size: int = REFERENCE_PARLAY_SIZE) -> float:
    """Implied per-leg decimal odds from the book's reference POWER payout.

    PrizePicks 4-pick power pays 10x -> 10 ** 0.25 = 1.778 per leg.
    Underdog   4-pick power pays 10x in the shipped ladder (12x on some boards);
    the FAQ quotes 1.86, which corresponds to 12x. We use the ladder we ship.
    """
    net = ladder(book, "POWER", reference_size)[-1]
    return (net + 1.0) ** (1.0 / reference_size)


def available_slips(book: str) -> list[tuple[str, int]]:
    out = []
    for slip_type, sizes in LADDERS[book.lower()].items():
        for n in sizes:
            out.append((slip_type, n))
    return out
