# edgeline

An open, self-hosted version of the LCSLarry-style esports player-prop model: capture
pick'em lines the minute they post, price every prop with a calibrated per-map count
model, compute expected value against the book's payout ladder, price correlated
stacks, auto-build slips, alert on Discord, and grade every prediction against both the
opening and closing line.

The research that motivated the design is in `research/lcslarry-esports-model-report.md`.

## Feature parity with the original product

| LCSLarry feature | edgeline | Notes |
|---|---|---|
| Scrape lines every minute, opening line tracking | `lines pull` / `lines watch` | PrizePicks via its partner API host (no bot wall). Every poll appends a snapshot; first sighting = opening line. |
| 13 books | 1 (PrizePicks) | Underdog client is a stub that needs the app's client headers (`EDGELINE_UNDERDOG_HEADERS`). |
| 5 esports | Dota 2, CS2, Valorant loaders; LoL loaders written | OpenDota (Dota), bo3.gg (CS2), vlr.gg (Valorant). Leaguepedia and Oracle's Elixir loaders exist for LoL but both throttled the research container. COD: no reachable source yet. |
| Per-prop projection + hit probability | `predict` | LightGBM Poisson per map, NB dispersion, isotonic calibration. |
| Multi-map and combo props | `predict` | Gaussian copula over (player, map) components with self / teammate / opponent correlations estimated from residuals. |
| EV vs fixed payouts, 60% / 5% EV thresholds, demon/goblin excluded, voidable filter, bumped-line rule | `predict` | Same defaults as the original. |
| Correlated stacks page | `stacks` | Joint hit probability of two same-game legs, lift over independence, break-even 2-pick multiplier to compare with the app's shaded payout. |
| Slip builder with one-click tail links | `slips build` | Greedy EV-optimal, exposure limits, exact Poisson-binomial slip EV, PrizePicks deep links, used-line tracking. |
| Discord line-drop alerts | `alerts send`, `lines watch --alert` | One alert per bettable projection. |
| Data-driven player ban list | `banlist update` | One-sided binomial test of graded picks vs the model's own probabilities. |
| Public results tracker (open vs close, 4-pick ROI) | `grade`, `results` | Same conventions as the original tracker. |
| EV calculator + bankroll simulator | `ev table`, `ev simulate` | Payout ladders lifted from the original calculator. |
| Web dashboard | not built | Everything is CLI + SQLite; a UI can sit on the same tables. |

## Quick start

```bash
pip install -e .            # Python 3.10+
edgeline init
edgeline lines pull --sports lol,cs2,val,dota,cod   # snapshot the boards (paced; ~1 min)

# history (pick the sports you want)
edgeline history opendota --since 2025-09-25        # Dota 2: ~25 s, ~90k pro player-match rows
edgeline history bo3 --since 2026-07-27             # CS2: one request per map, ~1 h for 60 days
edgeline history vlr --pages 40                     # Valorant: ~1.5 s per match, ~1 h for 5 months
edgeline history stats

edgeline model train --sport dota                   # kills, deaths, assists (+ headshots where data exists)
edgeline predict --sport dota --show-all            # price the board, star bettable lines
edgeline stacks --sport dota                        # correlated same-game pairs
edgeline slips build --sports dota --size 3         # EV-optimal 3-pick powers with tail links
edgeline alerts send --dry-run                      # what would be posted to Discord

# later, once matches finish:
edgeline history opendota --since <yesterday>
edgeline grade --sport dota
edgeline results --by sport
edgeline banlist update
```

Keep the opening-line capture running (the edge is at open), with alerts:

```bash
export EDGELINE_DISCORD_WEBHOOK=https://discord.com/api/webhooks/...
edgeline lines watch --sports lol,cs2,val,dota,cod --interval 90 --alert
```

## First real run (2026-09-25), Dota 2

One year of OpenDota history (92,710 player-match rows, 9,271 matches), time-split validation
on the last 20% of dates (from 2026-05-16):

| Stat | MAE model | MAE player last-10 mean | MAE global mean | NB r | rho self / team / opp |
|---|---|---|---|---|---|
| kills | 2.945 | 3.098 | 3.499 | 3.26 | 0.09 / 0.29 / -0.05 |
| deaths | 2.567 | 2.647 | 2.881 | 5.46 | 0.12 / 0.53 / -0.10 |
| assists | 5.684 | 5.939 | 6.100 | 3.61 | 0.13 / 0.75 / 0.00 |

Calibration after isotonic fitting is flat across buckets (kills: predicted 0.63 vs observed 0.63
in the 0.6 to 0.7 bucket, n=6,309).

Two backtest policies are printed by `model train`:

- Against lines placed at the model's own mean (optimistic): kills 66.2% hit rate at >= 60%.
- Against a naive book that sets every line at the player's trailing 10-game mean: kills 66.7%
  at >= 60% (n=7,161), 69.6% at >= 65%.

On the live PrizePicks Dota board that day, 37 of 40 lines priced and 8 cleared the defaults.
The stacks pricer found a teammate same-direction pair with a 17% lift over independence
(break-even 2.58x vs the standard 3x two-pick payout).

Read these honestly:

- Neither backtest uses real book lines. Real lines are sharper than a trailing mean, so
  expect a lower realized hit rate. The only number that matters is the graded hit rate on
  real opening lines, which accrues as you run `lines watch` and `grade` over weeks.
- The model has no moneyline odds, no draft/hero data and no map-pick data. Those are the
  likeliest sources of the original's extra edge and are the next features to add.
- Pushes on integer lines refund the stake; the pricer accounts for that. Inside a slip the
  builder treats a push as a loss (conservative).
- `MAPS 1-3` props are flagged voidable (best-of-3 assumed) and skipped by default. Verify
  PrizePicks' current void rules.
- Correlated stacks combine legs priced by the same stat model only; kills-vs-deaths pairs
  are treated as independent for now.

## How pricing works

1. `features.build` turns `player_games` into as-of rows: EWM and rolling means/stds of the
   player's kills/deaths/assists/headshots, kill share, game length, days since last game;
   team kills, kills conceded, win rate; the same for the opponent; map number; playoffs;
   role, league and tier as categoricals. Every value uses only prior games (shift-then-roll).
2. `models.props.train` fits a LightGBM Poisson model for the per-map mean, fits the NB
   dispersion `r` on held-out games, estimates residual correlations between a player's maps
   (rho_self), teammates (rho_team) and opponents (rho_opp), and fits isotonic calibration on
   tail probabilities at lines near the mean.
3. `models.predict.BoardPricer` resolves book names (players by normalized name; team codes by
   majority vote of their players' latest team), builds one (player, map) component per line,
   predicts the mean per component, and prices single components analytically or multi-
   component lines by Gaussian-copula simulation (`models.copula`).
4. EV per leg uses the book's implied per-leg odds from the 4-pick power ladder (PrizePicks
   10x -> 1.778 per leg, break-even 56.2%), with pushes as refunds.
5. `models.stacks` reuses the same components to price two legs jointly.
6. `slips.builder` ranks bettable legs by EV, enforces exposure limits, and prices each slip
   with the exact Poisson-binomial over the ladder (`ev.math`).

## Layout

```
edgeline/
  books/       prizepicks.py (client + open/current tracking), underdog.py (stub), normalize.py
  data/        opendota.py (Dota), bo3.py (CS2), vlr.py (Valorant), leaguepedia.py + oracles_elixir.py (LoL)
  features/    build.py       as-of feature engineering
  models/      props.py (train/calibrate), distributions.py (NB), copula.py (joint sims),
               predict.py (BoardPricer), stacks.py, banlist.py
  ev/          payouts.py (ladders), math.py (leg/slip EV, break-even, Kelly), simulate.py
  slips/       builder.py
  grading/     grade.py (settlement), results.py (tracker-style summary)
  alerts.py, db.py, config.py, cli.py
tests/         29 tests
```

## Roadmap

1. LoL history from a machine Leaguepedia has not throttled, then `model train --sport lol`.
2. COD (Breaking Point) and Underdog headers; then ParlayPlay, Dabble, Sleeper.
3. Match moneyline odds and map picks as features.
4. Cross-stat correlations (kills vs deaths) for stacks.
5. A small web dashboard over the SQLite tables.

## Disclaimer

Research software. Gambling involves risk; nothing here guarantees profit. Scraping book
endpoints may violate their terms of service. Check the laws in your jurisdiction.
