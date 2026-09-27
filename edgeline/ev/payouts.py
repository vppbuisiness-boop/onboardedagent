"""Payout ladders for fixed-payout pick'em books.

Each ladder is indexed by the number of legs that hit and gives the NET profit
per 1 unit staked (so -1 is a total loss, 9 means the slip returned 10x).

The PrizePicks and Underdog ladders below are the exact arrays shipped in
LCSLarry's public EV calculator (ev.lcslarry.com) as of September 2026.
Books change these from time to time; verify against the app before relying
on a ladder for real money.
"""
from __future__ import annotations

import os
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
        3: [-1, -1, 0.0, 2.0],  # 3 of 3 = 3x, 2 of 3 = 1x (prizepicks.com payouts page, 2026-09-27)
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

# Sleeper prices every pick with its own multiplier (1.78 standard, 1.6-1.9 when a line is shaded) and a slip pays
# the product of its legs' multipliers; the ladder below is the standard-multiplier case, and per-line odds
# override it at pricing time.
SLEEPER = {"POWER": {n: [-1.0] * n + [round(1.78 ** n - 1.0, 3)] for n in range(2, 7)}}
LADDERS = {"prizepicks": PRIZEPICKS, "underdog": UNDERDOG, "sleeper": SLEEPER}

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


# Pick'em Arena: PrizePicks' peer-to-peer format (August 2025, now the default in most of its states). An entry is
# pooled with others of the same size and type; it pays the standard multiplier only for first place in its group and
# otherwise a lower minimum guarantee when the picks hit. The API carries no payout data, so the guarantees are read
# off the app's entry card (net return per $1 by hits); sizes not listed fall back to the standard ladder and the slip
# chooser flags them unverified. 3-pick card, 2026-09-27: Power 3.5x guaranteed (6x for first place), Flex 2.75x on 3
# of 3 and 0.5x on 2 of 3 (3x for first place). Set EDGELINE_PRIZEPICKS_FORMAT=standard for an account on classic Pick'em.
ARENA_GUARANTEES: dict[str, dict[int, list[float]]] = {
    "POWER": {3: [-1, -1, -1, 2.5]},
    "FLEX": {3: [-1, -1, -0.5, 1.75]},
}
PRIZEPICKS_FORMAT = os.environ.get("EDGELINE_PRIZEPICKS_FORMAT", "arena").lower()
SPORT_LADDERS: dict[str, dict[str, dict[str, dict[int, list[float]]]]] = {}


def arena_guarantee(slip_type: str, size: int) -> list[float] | None:
    return list(ARENA_GUARANTEES.get(slip_type.upper(), {}).get(int(size), [])) or None


def ladder(book: str, slip_type: str, size: int, sport: str | None = None) -> list[float]:
    """The net ladder the account is paid on: the Arena minimum guarantee where the card has been read, else the
    book's standard ladder (which is also Arena's first-place payout)."""
    if book.lower() == "prizepicks" and PRIZEPICKS_FORMAT == "arena":
        g = arena_guarantee(slip_type, size)
        if g is not None:
            return g
    if sport:
        by_sport = SPORT_LADDERS.get(book.lower(), {}).get(sport.lower(), {})
        if int(size) in by_sport.get(slip_type.upper(), {}):
            return list(by_sport[slip_type.upper()][int(size)])
    try:
        return list(LADDERS[book.lower()][slip_type.upper()][int(size)])
    except KeyError as exc:
        raise KeyError(f"no ladder for book={book} type={slip_type} size={size}") from exc


def ladder_sport(legs: list[dict], book: str, slip_type: str, size: int) -> str | None:
    """The sport whose league-specific ladder applies to a slip: the app prices a mixed lineup at its reduced ladder
    when any leg belongs to a reduced-payout league, so the lowest top payout among the legs' sports wins."""
    best: tuple[float, str] | None = None
    for sp in {str(l.get("sport", "")).lower() for l in legs if l.get("sport")}:
        if int(size) in SPORT_LADDERS.get(book.lower(), {}).get(sp, {}).get(slip_type.upper(), {}):
            top = SPORT_LADDERS[book.lower()][sp][slip_type.upper()][int(size)][-1]
            if best is None or top < best[0]:
                best = (top, sp)
    return best[1] if best else None


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
