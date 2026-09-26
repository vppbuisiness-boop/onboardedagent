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

Eighteen months of bo3.gg history (329,622 player-map rows, 32,987 maps, 4,313 players, tiers S to C,
since 2025-04-01), validation on the last 20% of dates (from 2026-06-12):

| Stat | MAE model | MAE player last-10 mean | MAE global mean | NB r | rho self / team / opp |
|---|---|---|---|---|---|
| kills | 4.209 | 4.417 | 4.320 | 13.7 | 0.04 / 0.37 / 0.28 |
| deaths | 3.067 | 3.307 | 3.168 | 67.7 | 0.07 / 0.81 / 0.48 |
| assists | 1.991 | 2.075 | 2.057 | 12.0 | 0.02 / 0.21 / 0.16 |
| headshots | 2.702 | 2.803 | 2.915 | 13.4 | 0.02 / 0.18 / 0.15 |

The residual correlations reproduce the original product's stacking rules from data: in the
MOBA, teammates move together and opponents move slightly against each other; in the
round-based shooter every player in the match moves together (more rounds, more of everything),
so same-direction stacks are the only ones worth pricing.

### Valorant

Fourteen months of vlr.gg history (86,492 player-map rows, 8,650 maps, 2,905 players, all events,
since 2025-08-01), validation on the last 20% of dates (from 2026-06-25):

| Stat | MAE model | MAE player last-10 mean | MAE global mean | NB r | rho self / team / opp |
|---|---|---|---|---|---|
| kills | 4.116 | 4.300 | 4.274 | 17.3 | 0.02 / 0.24 / 0.16 |
| deaths | 2.779 | 2.953 | 2.874 | 200.0 | 0.06 / 0.76 / 0.31 |
| assists | 2.408 | 2.477 | 2.691 | 8.6 | 0.01 / 0.14 / 0.07 |

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

### First graded results on real opening lines (2026-09-25)

Settled the same day, graded against the opening line (`edgeline roi`, 21:59 UTC pass):

| Slice | Record | Hit rate (95% CI) | 4-pick ROI at the point estimate |
|---|---|---|---|
| All model leans | 109-91 | 54.5% (47.6% to 61.3%) | -12% |
| Bettable picks (>= 60% prob, >= 5% EV) | 20-14 | 58.8% (42.2% to 73.6%) | +20% |
| CS2 leans | 81-82 | 49.7% (42.1% to 57.3%) | -39% |
| Dota leans | 28-9 | 75.7% (59.9% to 86.6%) | +228% |

Two hundred lines prove little, but two things already match the corrected backtest: CS2 leans are a coin
flip (which is why CS2 kills is no longer flagged bettable), and Dota leans are winning, with the interval's
lower bound above the 56.2% break-even for the first time. The bettable record still contains the CS2 kills
picks flagged before the market gate. The model leans UNDER on most lines (tomorrow's LoL board: 63 of 65
priced lines); the model's kill levels check out league by league, so that is the book's lines sitting above
the model's mean, a pattern to be settled by grading rather than a finding.

### Backtest policies (printed by `model train`)

Validation-split hit rates against a naive book whose line is the median-fair half nearest the player's
trailing 10-game mean (COD: per game mode), after the drift-aware calibration. The mean-bias column is the
held-out ratio mean(actual) / mean(predicted) over the most recent 90 days that `model train` folds into
every projection.

| Sport / stat | mean bias | >= 60% | >= 65% |
|---|---|---|---|
| Dota kills | 1.044 | 66.1% (n=24,346) | 70.1% (n=13,943) |
| Dota deaths | 1.062 | 66.7% (n=25,303) | 70.4% (n=14,604) |
| Dota assists | 1.042 | 66.5% (n=23,446) | 70.4% (n=12,951) |
| CS2 kills | 29,371, 68.5%, +120% | 2,523 (2.7%), 64.2% (62.4% to 66.1%), +70% (+51% to +91%) | 58.5% on 865 (six months of history); 64.3% on 1,952 (twelve) |
| CS2 deaths | 1.019 | 68.8% (n=12,162) | 72.4% (n=8,111) |
| CS2 assists | 1.022 | 66.5% (n=9,684) | 70.6% (n=4,964) |
| CS2 headshots | 24,968, 68.0%, +114% | 3,395 (3.7%), 64.7% (63.1% to 66.3%), +76% (+59% to +94%) | 60.4% on 3,157 (six months of history); 64.2% on 2,781 (twelve) |
| VAL kills | 0.988 | 65.9% (n=3,697) | 70.2% (n=1,951) |
| VAL deaths | 0.990 | 67.3% (n=4,577) | 70.3% (n=2,963) |
| VAL assists | 0.967 | 64.6% (n=3,364) | 67.5% (n=1,730) |
| LoL kills | 0.997 | 67.6% (n=7,866) | 71.6% (n=4,449) |
| LoL deaths | 0.999 | 67.4% (n=8,905) | 70.8% (n=5,378) |
| LoL assists | 1.007 | 67.9% (n=8,300) | 71.3% (n=4,546) |
| COD kills (mode-aware naive line) | 1.009 | 63.2% (n=2,769) | 64.5% (n=1,486) |
| COD deaths (mode-aware naive line) | 1.010 | 65.6% (n=3,423) | 69.7% (n=2,263) |
| COD assists (mode-aware naive line) | 1.052 | 64.4% (n=3,409) | 67.3% (n=2,016) |

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

### Which markets are flagged bettable

Every posted line is priced and graded so the record keeps growing, but a line is flagged bettable only
in markets the evidence supports: the walk-forward backtest at the 60% threshold against a fair book-like
setter clears break-even with its whole interval for Dota, Valorant, LoL, CS2 headshots and, since the
twelve-month history load, CS2 kills (64.3%, interval 62.2% to 66.4%). COD (57.5%) has an interval that
includes losing money and carries the note `market_unproven`; it is excluded from slips unless
`edgeline predict --include-unproven` is used (`UNPROVEN_MARKETS` in `config.py`). CS2 kills was gated the
same way for the evening of 2026-09-25, when the six-month model backtested at 58.5% and ran 37-46 on
captured lines; the gate lifts with the twelve-month model and the captured-line record decides whether it
stays lifted. A 4-pick power slip
turns a per-leg rate into ROI by the fourth power: 56.2% is break-even, 58.5% is +17%, 60.4% is +33%,
66.4% is +94%, so a market two points above break-even is not a small step down from one ten points
above it.

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

Hit rates at the >= 60% threshold, with 95% Wilson intervals and the 4-pick POWER ROI they imply. The
last column is the same run before the drift-aware calibration described below:

| Sport | vs naive: picks, hit rate, ROI | vs book-like: picks (share of games), hit rate (CI), ROI (CI) | before drift fix |
|---|---|---|---|
| LoL | 10,377, 69.0%, +127% | 1,690 (5.1%), 63.6% (61.3% to 65.9%), +64% (+41% to +88%) | 64.2% on 1,627 |
| Dota 2 | 5,836, 68.1%, +115% | 1,045 (5.9%), 66.4% (63.5% to 69.2%), +95% (+63% to +129%) | 63.3% on 1,639 |
| Valorant | 7,859, 67.9%, +113% | 541 (1.8%), 65.6% (61.5% to 69.5%), +85% (+43% to +133%) | 64.5% on 558 |
| CS2 kills | 28,051, 67.0%, +102% | 865 (1.0%), 58.5% (55.2% to 61.7%), +17% (-7% to +45%) | 58.1% on 880 |
| CS2 headshots | 26,722, 65.9%, +89% | 3,157 (3.5%), 60.4% (58.7% to 62.1%), +33% (+18% to +48%) | 59.6% on 3,030 |
| COD | 4,326, 66.5%, +95% | 731 (5.5%), 57.5% (53.8% to 61.0%), +9% (-16% to +38%) | 57.6% on 785 |

Twelve months of CS2 history: bo3.gg maps from October 2025 to March 2026 were added (233,768 player-maps
in total, 3,840 players) and the CS2 kills row above is that run; with six months the same setter gave
58.5% on 865 picks, so the extra history is what moved CS2 kills over the 60% bar (more players clear the
three-game minimum and every form window is longer). CS2 headshots moved the same way: 60.4% on 3,157 picks
with six months, 64.2% on 2,781 (interval 62.4% to 65.9%) with twelve. Eighteen months (April 2025 on, 329,622
player-maps) kept the accuracy and added 22% to 29% more qualifying picks (kills 64.2% on 2,523, headshots
64.7% on 3,395), so the table shows the eighteen-month runs and the live CS2 models use that history. Both CS2
markets now clear the 60% bar in backtest; their captured-line record with these models starts on 2026-09-26.

Valorant history was extended the same way (August 2025 on, 86,492 player-maps, up from seven months) and the
result cuts the other way against the book-like setter: 61.2% on 273 picks (interval 55.3% to 66.8%) versus
65.6% on 541 with seven months, while the model's own error fell (fold MAE 4.126 vs 4.151) and the naive
comparison rose (68.9% vs 67.8%). The book-like setter is trained on the same history, so more data sharpens
it too and the model disagrees with it less often; a real book does not improve because this repository
loaded more data. The live Valorant models use the fourteen months (better error); the table keeps the
seven-month book-like figure because it is the larger sample, with the fourteen-month figure noted here.

Drift-aware calibration: the Poisson GBM's means ran 1% to 4% low out of sample (Dota 3%, CS2 headshots 3%),
and the level of kills drifts within a season (Dota fell from 5.9 to 5.0 per game over the 2025-26 winter and
climbed back to 5.7 by September). A calibrator fit across the whole held-out split learned the average of
that drift and pushed every probability toward UNDER in the months that followed (Dota: predicted P(over)
0.52, observed 0.60; 92% of picks were unders). `model train` now estimates the mean-bias factor, the
dispersion and the probability calibrator on the most recent 90 days of the held-out split. Picks are
now balanced enough to trust both sides (share of overs among book-like picks: LoL 16%, Dota 2 24%, Valorant 63%, CS2 kills 35%, CS2 headshots 35%, COD 48%), and overs and unders
hit at the same rate in Dota (66.5% / 66.4%). CS2's split is only five weeks long, so there the change is the
mean-bias factor alone.

Raising the threshold to 65% against the book-like setter gives LoL 65.8% (n=120), Dota 2 61.1% (n=90), CS2 kills 56.7% (n=67), CS2 headshots 66.8% (n=211), COD 54.7% (n=203); the other sports have too few
picks above 65% to read. Earlier feature decisions (map-pool expectation for CS2 and COD, two years of
Dota history, the 2025 LoL season, and the rejected round-count features) were made from comparisons
under the previous setter and have not been re-measured under this one. Two feature variants were
measured under the corrected setter (before the drift fix) and left off by default: bo3.gg per-map ADR,
KAST, first kills/deaths and rating for CS2 (`EDGELINE_EXTRA_STATS=1`: kills 57.0% on 944 picks vs 58.1%,
headshots 59.8% on 3,359 vs 59.6%, identical error) and round-count features for Valorant
(`EDGELINE_ROUND_FEATURES=1`: 65.4% on 534 picks vs 64.5%, within noise).

How to read it:

- Against a naive line-setter every sport clears the 60% bar (the 30% ROI bar) by 6 to 9 points, with
  overs and unders hitting at similar rates. The model is far better than a trailing average.
- Against a book that prices from the same public averages, Dota, Valorant, LoL, CS2 kills and CS2
  headshots clear 60% with the whole interval above it, on only 2% to 6% of games (the model rarely
  disagrees with a fair line by enough). COD sits at 58%: above the 56.2% break-even at the point
  estimate, but its interval includes losing money. In the round-based shooters,
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
