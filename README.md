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
| 5 esports | all five have working loaders: Dota 2, CS2, Valorant, LoL, COD | OpenDota (Dota), bo3.gg (CS2), vlr.gg (Valorant), Riot's official esports API + livestats feed (LoL), Breaking Point's public database (COD). Leaguepedia / Oracle's Elixir loaders remain as alternatives. |
| Per-prop projection + hit probability | `predict` | LightGBM Poisson per map, NB dispersion, logistic recalibration, market-prior shrink toward the book line. |
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
edgeline history lolesports --since 2026-01-01      # LoL: Riot's esports API, all major leagues, ~20 min
edgeline history breakingpoint --sport cod --since 2025-10-28   # COD: one season in ~1 min
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

## First real runs (2026-09-25)

### Dota 2

One year of OpenDota history (92,710 player-match rows, 9,271 matches), time-split validation
on the last 20% of dates (from 2026-05-16):

| Stat | MAE model | MAE player last-10 mean | MAE global mean | NB r | rho self / team / opp |
|---|---|---|---|---|---|
| kills | 2.945 | 3.098 | 3.499 | 3.26 | 0.09 / 0.29 / -0.05 |
| deaths | 2.567 | 2.647 | 2.881 | 5.46 | 0.12 / 0.53 / -0.10 |
| assists | 5.684 | 5.939 | 6.100 | 3.61 | 0.13 / 0.75 / 0.00 |

### CS2

Sixty days of bo3.gg history (44,380 player-map rows, 4,447 maps, tiers S to C), validation on
the last 20% of dates:

| Stat | MAE model | MAE player last-10 mean | MAE global mean | NB r | rho self / team / opp |
|---|---|---|---|---|---|
| kills | 4.154 | 4.320 | 4.227 | 14.4 | 0.11 / 0.34 / 0.23 |
| deaths | 2.951 | 3.168 | 3.026 | 116 | 0.12 / 0.79 / 0.43 |
| assists | 2.002 | 2.053 | 2.039 | 12.2 | 0.09 / 0.20 / 0.14 |
| headshots | 2.675 | 2.747 | 2.858 | 14.6 | 0.08 / 0.15 / 0.11 |

The residual correlations reproduce the original product's stacking rules from data: in the
MOBA, teammates move together and opponents move slightly against each other; in the
round-based shooter every player in the match moves together (more rounds, more of everything),
so same-direction stacks are the only ones worth pricing.

### Valorant

Seven months of vlr.gg history (50,212 player-map rows, 5,022 maps, all events), validation on
the last 20% of dates:

| Stat | MAE model | MAE player last-10 mean | MAE global mean | NB r | rho self / team / opp |
|---|---|---|---|---|---|
| kills | 4.104 | 4.288 | 4.257 | 16.9 | 0.05 / 0.24 / 0.14 |
| deaths | 2.786 | 2.942 | 2.861 | 200 | 0.12 / 0.76 / 0.26 |
| assists | 2.381 | 2.427 | 2.659 | 9.2 | 0.02 / 0.15 / 0.07 |

Headshot counts are not published on vlr.gg, so Valorant headshot props are not priced.

### Backtest policies (printed by `model train`)

| Sport / stat | vs lines at the model's own mean, >= 60% | vs a naive book (line = trailing 10-game mean), >= 60% | >= 65% |
|---|---|---|---|
| Dota kills | 66.2% (n=14,960) | 65.9% (n=7,930) | 69.4% |
| Dota deaths | 64.9% | 64.9% (n=6,624) | 67.6% |
| CS2 kills | 62.1% | 65.9% (n=3,349) | 69.3% |
| CS2 deaths | 65.6% | 68.8% (n=4,245) | 72.3% |
| CS2 headshots | | 64.4% (n=3,427) | 68.3% |
| VAL kills | | 65.6% (n=4,079) | 69.6% |
| VAL deaths | | 66.7% (n=4,459) | 69.6% |

Read these honestly:

- Neither backtest uses real book lines. Real lines are sharper than a trailing mean, so
  expect a lower realized hit rate. The only number that matters is the graded hit rate on
  real opening lines, which accrues as you run `lines watch` and `grade` over weeks.
- The CS2 model beats the naive baselines by only 2 to 5% MAE; per-map kills there are mostly
  a function of rounds played, which the model sees only through team strength proxies.
- The model has no moneyline odds, no draft/hero data and no map-pick data. Those are the
  likeliest sources of the original's extra edge and are the next features to add.
- By default the pricer shrinks each component's projection 25% toward the book's implied
  per-map line (`--market-shrink`). It is the conservative direction when a short-history
  model disagrees sharply with a book; tune it once grades accumulate.
- Pushes on integer lines refund the stake; the pricer accounts for that. Inside a slip the
  builder treats a push as a loss (conservative).
- `MAPS 1-3` props are flagged voidable (best-of-3 assumed) and skipped by default. Verify
  PrizePicks' current void rules.
- Correlated stacks combine legs priced by the same stat model only; kills-vs-deaths pairs
  are treated as independent for now.
- On the live board that day: Dota 37 of 40 lines priced (8 bettable); CS2 284 of 393 priced,
  60 bettable after shrinkage, 109 unpriced because the player had no maps in the 60-day window
  (extend `history bo3 --since` to cover them); Valorant 78 of 78 priced, 11 bettable, combos included.

## How pricing works

1. `features.build` turns `player_games` into as-of rows: EWM and rolling means/stds of the
   player's kills/deaths/assists/headshots, kill share, game length, days since last game;
   team kills, kills conceded, win rate; the same for the opponent; map number; playoffs;
   role, league and tier as categoricals. Every value uses only prior games (shift-then-roll).
2. `models.props.train` fits a LightGBM Poisson model for the per-map mean, fits the NB
   dispersion `r` on held-out games, estimates residual correlations between a player's maps
   (rho_self), teammates (rho_team) and opponents (rho_opp), and fits a logistic recalibration
   on tail probabilities at lines spread four counts either side of the mean, for single maps
   and two-map sums. (An isotonic fit was tried first; it only covered lines near the mean and
   clipped everything outside to 0 or 1, which produced fake 99.9% legs.)
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
tests/         35 tests
```

## How each data block was fixed

| Block | What failed | Fix |
|---|---|---|
| LoL history | Leaguepedia's Cargo API rate-banned the IP for hours after one burst; Oracle's Elixir CSVs hit Google Drive's download quota | Riot's own esports API (`esports-api.lolesports.com`, public site key) lists leagues, tournaments and completed matches; the livestats feed (`feed.lolesports.com/livestats/v1/window/{gameId}`) returns each game's final frame with kills, deaths and assists per player when asked for a time after the game ended. No key, no throttling seen at 6 concurrent workers. |
| COD history | breakingpoint.gg is a client-rendered app with no visible API | Its Next.js bundle ships a Supabase anon key and the stats tables allow anonymous reads (`player_stats`, `games`, `matches`, `teams`, `maps`, `modes`), so one paged REST query per table covers a season. Also holds CS2 stats (`sport_id` 2). |
| CS2 history | HLTV is Cloudflare-walled for scripts and headless browsers | bo3.gg's JSON API (`api.bo3.gg/api/v1`) exposes matches, per-map games and `games/{id}/players_stats`. Its date filters are ignored, so the loader pages newest-first to the cutoff; per-map calls run in a thread pool because latency is spiky. Missing tier-1 players were a window problem: load 6 months, not 60 days. |
| Underdog lines | `api.underdogfantasy.com` answers 426 unless the request carries the web app's current client headers, and the app itself sits behind a bot wall | Capture once in a browser: open underdogfantasy.com, DevTools > Network, click any request to api.underdogfantasy.com, copy `client-type`, `client-version`, `client-device-id` (and the User-Agent) into `EDGELINE_UNDERDOG_HEADERS` as JSON. Then `edgeline lines underdog-dump` saves the raw JSON; the line parser is written against that file. Guessed versions do not work, the value is a build identifier. |
| PrizePicks throttling | Bursts of 3+ requests get HTTP 429 from Cloudflare | Requests are spaced 12 s per league with exponential backoff; one full cycle of five boards takes about a minute, which is fine for a 90 s watch loop. |
| Book line history | You cannot backtest line timing without your own captured lines | `lines watch` appends a snapshot on every poll; the first sighting is the opening line. Start it now on a machine that stays up. |

## Roadmap

1. Underdog parser once headers are captured; then ParlayPlay, Dabble, Sleeper.
2. Match moneyline odds and map picks as features.
3. Cross-stat correlations (kills vs deaths) for stacks.
4. A small web dashboard over the SQLite tables.

## Disclaimer

Research software. Gambling involves risk; nothing here guarantees profit. Scraping book
endpoints may violate their terms of service. Check the laws in your jurisdiction.
