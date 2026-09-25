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
| 5 esports | all five trained: Dota 2, CS2, Valorant, LoL, COD | OpenDota (Dota), bo3.gg (CS2), vlr.gg (Valorant), Riot's official esports API + livestats feed (LoL), Breaking Point's public database (COD). Leaguepedia / Oracle's Elixir loaders remain as alternatives. |
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
edgeline history lolesports --since 2025-01-01      # LoL: Riot's esports API, every league except TFT, ~40 min
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

Two years of OpenDota history (304,380 player-match rows, 30,438 matches), time-split validation
on the last 20% of dates (from 2025-11-23):

| Stat | MAE model | MAE player last-10 mean | MAE global mean | NB r | rho self / team / opp |
|---|---|---|---|---|---|
| kills | 2.863 | 3.003 | 3.552 | 3.35 | 0.05 / 0.27 / -0.07 |
| deaths | 2.425 | 2.532 | 2.845 | 6.17 | 0.07 / 0.53 / -0.11 |
| assists | 5.467 | 5.747 | 6.023 | 3.75 | 0.06 / 0.75 / -0.06 |

### CS2

Six months of bo3.gg history (118,096 player-map rows, 11,840 maps, 2,975 players, tiers S to C),
validation on the last 20% of dates:

| Stat | MAE model | MAE player last-10 mean | MAE global mean | NB r | rho self / team / opp |
|---|---|---|---|---|---|
| kills | 4.193 | 4.385 | 4.282 | 14.4 | 0.11 / 0.34 / 0.23 |
| deaths | 3.036 | 3.242 | 3.105 | 116 | 0.12 / 0.79 / 0.43 |
| assists | 1.985 | 2.067 | 2.040 | 12.2 | 0.09 / 0.20 / 0.14 |
| headshots | 2.682 | 2.781 | 2.904 | 14.6 | 0.08 / 0.15 / 0.11 |

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

### League of Legends

Riot's official esports API and livestats feed, every league Riot lists except TFT (38 leagues with
games) since 2025-01-13 (103,420 player-game rows, 10,342 games; LPL is thin because Riot's feed rarely
carries LPL livestats), time-split validation on the last 20% of dates (from 2026-07-16):

| Stat | MAE model | MAE player last-10 mean | MAE global mean | NB r | rho self / team / opp |
|---|---|---|---|---|---|
| kills | 1.970 | 2.090 | 2.368 | 3.27 | -0.00 / 0.17 / -0.06 |
| deaths | 1.646 | 1.731 | 1.745 | 10.57 | 0.00 / 0.43 / -0.12 |
| assists | 3.440 | 3.675 | 3.936 | 4.04 | -0.03 / 0.64 / -0.16 |

### Call of Duty

Breaking Point's database, BO7 season since 2025-11-18 (32,218 player-map rows, 4,004 maps).
Kills depend on the mode (Hardpoint, Search & Destroy, Overload), which the map order fixes, so
the model uses per-(player, mode) rolling features and a map-to-mode override at pricing time:

| Stat | MAE model | MAE player last-10 mean (all modes) | NB r | rho self / team / opp |
|---|---|---|---|---|
| kills | 4.148 | 9.629 | 36.5 | 0.05 / 0.29 / 0.24 |
| deaths | 3.234 | 9.436 | 200 | 0.09 / 0.75 / 0.44 |

### First graded results on real opening lines (2026-09-25, same day)

The Dota board captured and priced that morning settled the same afternoon: 19 lines graded
against the opening line, 14 wins, 4 losses, 1 push (bettable subset: 4 wins, 2 losses). CS2 and
Valorant lines from that morning were not priced before they started (the watcher now prices
every board right after each pull). Nineteen lines prove nothing; the number is listed because
it is the first one in this repository measured against a real book.

### Backtest policies (printed by `model train`)

Note: the policy hit rates below were computed against lines at the trailing mean rounded up to the next .5,
which overstates under-side edge (see the correction note in the historical backtest section). They are
refreshed with the median-fair rule at the next `model train`.

| Sport / stat | vs lines at the model's own mean, >= 60% | vs a naive book (line = trailing 10-game mean), >= 60% | >= 65% |
|---|---|---|---|
| Dota kills | 66.2% (n=14,960) | 67.1% (n=8,125) | 70.5% |
| Dota deaths | 64.9% | 65.6% (n=7,455) | 69.7% |
| CS2 kills | 62.1% | 67.0% (n=9,382) | 70.0% |
| CS2 deaths | 65.6% | 68.8% (n=11,641) | 72.4% |
| CS2 headshots | | 65.3% (n=9,232) | 68.9% |
| VAL kills | | 65.9% (n=4,049) | 70.1% |
| VAL deaths | | 66.5% (n=4,523) | 70.2% |
| LoL kills | | 69.1% (n=3,964) | 73.0% |
| LoL deaths | | 67.2% (n=3,781) | 70.9% |
| COD kills (mode-aware naive line) | | 63.8% (n=2,644) | 66.4% |
| COD deaths (mode-aware naive line) | | 64.9% (n=3,405) | 68.8% |

Feature set as of these numbers: player form (EWM and rolling means), per-role form, team form,
team Elo and Elo gap (a win-probability proxy for the missing moneyline), opponent form, and
role-matchup terms (what the opponent's player in the same role scores, what the opponent concedes
to that role). Adding Elo and role matchups moved Dota, LoL and COD up by roughly one point at the
60% threshold and left CS2 and Valorant unchanged.

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
- On the live board that day: Dota 37 of 40 lines priced (8 bettable); CS2 196 of 279 priced with
  six months of history (30 bettable; the 83 unpriced are lower-tier or differently spelled names);
  Valorant 78 of 78 priced (11 bettable, combos included); LoL 30 of 30 priced, 9 bettable once
  Riot's schedule confirmed the series was a best-of-5 (maps 1 to 3 cannot void).

## How pricing works

1. `features.build` turns `player_games` into as-of rows: EWM and rolling means/stds of the
   player's kills/deaths/assists/headshots, kill share, game length, days since last game; per-role
   form; team kills, kills conceded, win rate and Elo; the same for the opponent; the Elo gap and
   its implied win probability; the opponent's same-role player's kills and the opponent's kills
   conceded to that role; map number; playoffs; role, league and tier as categoricals. Every value
   uses only prior games (shift-then-roll, ratings updated after each game).
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
tests/         38 tests
```

## Historical backtest (walk-forward, kills, 2026-05 to 2026-09)

There is no public archive of historical PrizePicks esports lines (the Wayback Machine, archive.today
and arquivo.pt hold none; the PrizePicks API only serves the live board), so a historical ROI can
only be measured against synthetic line-setters. `edgeline model backtest` retrains month by month
on prior games only, prices every player-game in the month with the live rules (25% shrink toward
the line, calibrated NB tails, bet the side at >= 60%), and grades against two setters:

- naive: the player's trailing 10-game mean in the role
- book-like: a separate model fit on the same past data from the basic averages a small book would
  use (player last-10/20, opponent kills conceded, team pace, role, map number)

Both setters place the line at the median-fair half: of the two .5 lines around their mean, the one
whose over probability is closest to 50% under the fitted negative binomial. This matters. An earlier
version of this table used the mean rounded up to the next .5; kill counts are right-skewed, so that
line sat above the median and a blind UNDER beat it 53% to 58% of the time, which credited the model
with an edge that was only the rounding rule. Captured PrizePicks lines behave like the corrected
setters (blind unders run near 50% on settled lines), so the numbers below are the ones to trust.

Hit rates at the >= 60% threshold, with 95% Wilson intervals and the 4-pick POWER ROI they imply:

| Sport | vs naive: picks, hit rate, ROI | vs book-like: picks (share of games), hit rate (CI), ROI (CI) |
|---|---|---|
| LoL | 10,344, 68.9%, +126% | 1,627 (4.9%), 64.2% (61.9% to 66.5%), +70% (+47% to +96%) |
| Dota 2 | 6,112, 66.9%, +100% | 1,639 (9.2%), 63.3% (60.9% to 65.6%), +60% (+38% to +85%) |
| Valorant | 7,844, 67.8%, +111% | 558 (1.9%), 64.5% (60.5% to 68.4%), +73% (+34% to +119%) |
| CS2 kills | 28,024, 67.1%, +102% | 880 (1.0%), 58.1% (54.8% to 61.3%), +14% (-10% to +41%) |
| CS2 headshots | 26,320, 66.0%, +90% | 3,030 (3.3%), 59.6% (57.9% to 61.4%), +26% (+12% to +42%) |
| COD | 4,350, 66.4%, +95% | 785 (5.9%), 57.6% (54.1% to 61.0%), +10% (-14% to +38%) |

Raising the threshold to 65% against the book-like setter gives LoL 64.5% (n=107), Dota 2 70.1% (n=134), CS2 kills 57.5% (n=73), CS2 headshots 68.6% (n=210), COD 56.3% (n=190); the other sports have too few
picks above 65% to read. Earlier feature decisions (map-pool expectation for CS2 and COD, two years of
Dota history, the 2025 LoL season, and the rejected round-count features) were made from comparisons
under the previous setter and have not been re-measured under this one; the CS2 ADR/KAST/rating and
Valorant round-count variants are reported below as they finish.

How to read it:

- Against a naive line-setter every sport clears the 60% bar (the 30% ROI bar) by 6 to 9 points, with
  overs and unders hitting at similar rates. The model is far better than a trailing average.
- Against a book that prices from the same public averages, LoL, Dota and Valorant clear 60% with the
  whole interval above it, on only 2% to 9% of games (the model rarely disagrees with a fair line by
  enough). CS2 kills, CS2 headshots and COD sit at 58% to 60%: above the 56.2% break-even at the point
  estimate, but the intervals for kills and COD include losing money. In the round-based shooters,
  kills are mostly a function of rounds played, which any competent book captures.
- Real PrizePicks lines carry information this model lacks (moneyline, drafts, map vetoes), so the
  book-like column is closer to reality than the naive one. The captured-line record so far agrees:
  CS2 leans are near a coin flip, Dota leans are winning.
- This is edge over a modeled setter, not realized ROI. The only real ROI is `edgeline roi` on
  captured opening lines, which needs roughly 120 settled bettable picks before its interval means
  anything and roughly 380 to demonstrate a 60% leg rate.

## How each data block was fixed

| Block | What failed | Fix |
|---|---|---|
| LoL history | Leaguepedia's Cargo API rate-banned the IP for hours after one burst; Oracle's Elixir CSVs hit Google Drive's download quota | Riot's own esports API (`esports-api.lolesports.com`, public site key) lists leagues, tournaments and completed matches; the livestats feed (`feed.lolesports.com/livestats/v1/window/{gameId}`) returns each game's final frame with kills, deaths and assists per player when asked for a time after the game ended. No key, no throttling seen at 6 concurrent workers. |
| COD history | breakingpoint.gg is a client-rendered app with no visible API | Its Next.js bundle ships a Supabase anon key and the stats tables allow anonymous reads (`player_stats`, `games`, `matches`, `teams`, `maps`, `modes`), so one paged REST query per table covers a season. Also holds CS2 stats (`sport_id` 2). |
| CS2 history | HLTV is Cloudflare-walled for scripts and headless browsers | bo3.gg's JSON API (`api.bo3.gg/api/v1`) exposes matches, per-map games and `games/{id}/players_stats`. Its date filters are ignored, so the loader pages newest-first to the cutoff; per-map calls run in a thread pool because latency is spiky. Missing tier-1 players were a window problem: load 6 months, not 60 days. |
| Underdog lines | `api.underdogfantasy.com` answers 426 unless the request carries the web app's current client headers, and the app itself sits behind a bot wall | Capture once in a browser: open underdogfantasy.com, DevTools > Network, click any request to api.underdogfantasy.com, copy `client-type`, `client-version`, `client-device-id` (and the User-Agent) into `EDGELINE_UNDERDOG_HEADERS` as JSON. Then `edgeline lines underdog-dump` saves the raw JSON; the line parser is written against that file. Guessed versions do not work, the value is a build identifier. |
| PrizePicks throttling | Bursts of 3+ requests get HTTP 429 from Cloudflare | Requests are spaced 12 s per league with exponential backoff; one full cycle of five boards takes about a minute, which is fine for a 90 s watch loop. |
| Book line history | You cannot backtest line timing without your own captured lines | `lines watch` appends a snapshot on every poll; the first sighting is the opening line. Start it now on a machine that stays up. |


Grading notes: bo3.gg links clan tags to canonical teams with a lag, so `history bo3` re-fetches maps from
the last two days (`--refresh-days`) and the feature builder merges case variants of a team name; the
grader voids a later-map line only once the loaded maps show a decided series (three map wins, or two
map wins after six hours), because sources publish maps one at a time.

## Roadmap

1. Underdog parser once headers are captured; then ParlayPlay, Dabble, Sleeper.
2. Match moneyline odds and map picks as features.
3. Cross-stat correlations (kills vs deaths) for stacks.
4. A small web dashboard over the SQLite tables.

## Disclaimer

Research software. Gambling involves risk; nothing here guarantees profit. Scraping book
endpoints may violate their terms of service. Check the laws in your jurisdiction.
