# Edgeline: the whole model, handed off

Written 2026-09-27 02:30 UTC from the live database and the code on branch `claude/amazing-ritchie-q2np39`.
This is the map of everything the project holds. Every claim below is either in the code, in the exported
tables, or printed by a command named here, so a fresh session can verify any of it.

## 0. Where everything lives

| Thing | Where |
|---|---|
| Repository | https://github.com/vppbuisiness-boop/onboardedagent, branch `claude/amazing-ritchie-q2np39` (122 commits; this branch is the live line, not `main`) |
| Clone | `git clone -b claude/amazing-ritchie-q2np39 https://github.com/vppbuisiness-boop/onboardedagent.git` |
| Install | `pip install -e .` (Python 3.10+: pandas, numpy, scipy, scikit-learn, lightgbm, requests, typer, joblib; `pytest` for the tests). The CLI is `edgeline`. |
| Website feeds (rebuilt and pushed every 3 hours) | `https://raw.githubusercontent.com/vppbuisiness-boop/onboardedagent/claude/amazing-ritchie-q2np39/data/web/v1/esports-ev.json`, `.../esports-slips.json`, `.../esports-record.json` (public, `access-control-allow-origin: *`) |
| Drop-in page for oddsfloor.com | `web/esports.html`; integration notes `web/README.md` |
| Workbook | `data/exports/edgeline_model.xlsx` (board, bettable picks, slips, real-line record, closing-line value, backtests, stack backtest, graded lines, model reference) |
| Exports (the only data that cannot be re-downloaded) | `data/exports/*.csv`: lines, line_snapshots, predictions, grades, slips, used_lines, kalshi_quotes, kalshi_grades, alerts_sent, banned_players. `edgeline import` restores them into a fresh database. |
| Trained models | `models/artifacts/<sport>_<stat>.joblib` and `<sport>_<stat>_metrics.json` (15 models). Gitignored; retrain with `edgeline model train --sport <s>` after loading history. |
| Research report the design came from | `research/lcslarry-esports-model-report.md` |
| Tests | `tests/` (59 passing: `python3 -m pytest -q tests`) |
| Long-form narrative with every experiment | `README.md` (595 lines) |

## 1. What it is, and the target

An open, self-hosted version of the LCSLarry esports player-prop model: capture PrizePicks pick'em lines the
minute they post, price every prop with a calibrated per-map count model, compute expected value against the
book's payout ladder, price correlated stacks, build the best slip at every price, grade every prediction against
the opening and closing line, and publish the record.

The target is the original product's headline: about 30% ROI. On a PrizePicks 4-pick power (10x) that is a
per-leg hit rate of 60% (0.6^4 x 10 - 1 = 29.6%). Break-even on that ladder is a 56.2% leg. So every number in
the project is a per-leg hit rate on real captured lines, with an interval, against 56.2% and 60%.

The standard for "proven": a market's match-cluster 95% interval lower bound above 56.2% over at least ten
matches, AND a selection value (hit rate minus the blind same-side rate on the same lines, section 10) clearly above
zero. A high hit rate that equals the blind rate is the market's tilt, not the model. No market has met the standard
(section 11). The bettable slice is above both bars at the point estimate with a selection value of +2.6 points.

Rules the whole project follows: no tuning to a target, only verifiable changes; standard lines only (demon and
goblin alternates are MORE-only and pay their own ladders); every proposed bet is inside the horizon (today or
tomorrow, New York time) and seen on the book's latest pull; every slip carries players from at least two teams;
single-match lineups are priced on PrizePicks' contest guarantee, not the standard ladder.

## 2. The pipeline

```
books/prizepicks.py, books/sleeper.py   ->  line_snapshots (every poll), lines (open = first sighting, current = last)
data/*.py (per sport history)           ->  player_games (one row per player per map or game)
features/build.py, features/mappool.py  ->  as-of feature rows (shift-then-roll; nothing from the row's own game)
models/props.py                         ->  per (sport, stat): LightGBM Poisson mean, NB dispersion, correlations, calibrator
models/predict.py (BoardPricer)         ->  predictions (projection, P(over/under/push), lean, prob, EV, bettable, notes)
ev/payouts.py, ev/math.py               ->  ladders, per-leg odds, Poisson-binomial slip EV, break-even, Kelly
models/copula.py, models/stacks.py      ->  multi-map sums, combos, same-match stacks priced jointly
slips/builder.py, slips/stacks.py, slips/best.py, slips/horizon.py  ->  slips (best slip at every price, stakes)
grading/grade.py -> grades; grading/results.py, roi.py, clv.py, track.py  ->  the record
books/kalshi.py + models/winner.py      ->  kalshi_quotes, kalshi_grades (exchange winner markets)
export.py                               ->  data/exports; web.py -> data/web/v1 feeds; scripts/build_workbook.py -> xlsx
```

## 3. Repository layout, module by module

| Module | Lines | What it does |
|---|---|---|
| `edgeline/cli.py` | 717 | Typer CLI (section 15 lists every command). |
| `edgeline/config.py` | 41 | Paths (`EDGELINE_ROOT`, `EDGELINE_DATA_DIR`, `EDGELINE_DB`, `EDGELINE_ARTIFACTS`), `USER_AGENT`, `DEFAULT_MIN_PROB = 0.60`, `DEFAULT_MIN_EV = 0.05`, `DEFAULT_MAX_LINE_MOVE = 0.10`, `DEFAULT_MARKET_SHRINK = 0.25`, `LOCAL_TZ = "America/New_York"`, `SLIP_HORIZON_DAYS = 1`, `UNPROVEN_MARKETS = {("cs2", "kills")}` with the evidence in a comment. |
| `edgeline/db.py` | 194 | SQLite schema and session (section 15). |
| `edgeline/export.py` | 53 | Export and import of the tables that cannot be re-downloaded. |
| `edgeline/alerts.py` | 66 | Discord webhook alerts for newly bettable lines (`EDGELINE_DISCORD_WEBHOOK`), one alert per projection, recorded in `alerts_sent`. |
| `edgeline/web.py` | 116 | The three website feeds in oddsfloor's `opportunities` schema (section 16). |
| `edgeline/books/prizepicks.py` | 230 | PrizePicks board client on `partner-api.prizepicks.com` (no DataDome wall), JSON:API; league ids lol 121, cs2 265, val 159, dota 174, cod 145, r6 274; 12 s pacing per league with exponential backoff on 429; `parse_board`, `upsert_lines` (first sighting = opening line), `tail_link` deep links `https://app.prizepicks.com/?projections=<id>-<o|u>-<line>,...`. |
| `edgeline/books/sleeper.py` | 126 | Sleeper Picks client (`api.sleeper.app`: `/lines/available?dynamic=true`, `/players/{sport}`, `/schedule/{sport}/regular/{year}`); CS kills and headshots maps 1-2, per-side multipliers (1.78 standard, 1.5 to 2.1 when shaded) stored as odds so EV uses the real payout. |
| `edgeline/books/underdog.py` | 34 | Stub: the API answers 426 without the web app's `client-type`, `client-version`, `client-device-id` headers (`EDGELINE_UNDERDOG_HEADERS` as JSON); `lines underdog-dump` saves the payload once they are supplied. |
| `edgeline/books/kalshi.py` | 241 | Kalshi esports map/game/series winner markets (`api.elections.kalshi.com/trade-api/v2`, public reads); `scan` records best yes bid/ask, volume and our probability per ticker; `grade` settles against loaded history and scores the model against the Kalshi mid (log-loss, Brier) and a buy-the-edge trade rule at 3, 5 and 10 cents. |
| `edgeline/books/normalize.py` | 101 | Stat-type parser: `MAP n` / `MAPS a-b` + stat (+ `(combo)`), `STAT_ALIASES` (kills, deaths, assists, headshots, fantasy, kills_assists, cs, last_hits, gpm, xpm, tower_kills, damage, first_bloods), `sport_from_league`, `opponent_from_game_id`, `voidable(sport, scope, series_format)` (a map range that can end early in the series). |
| `edgeline/books/series_format.py` | 98 | Best-of formats for upcoming LoL matches from Riot's schedule feed for every league (cache `data/raw/lol_formats.json`, 30 min TTL); `match_format(events, start_time, codes, names, window_minutes=90)` matches by team code, resolved name, or time; `voidable_with_format`. |
| `edgeline/books/shop.py` | 53 | Line shopping across books: same player and stat, each book's line, lean and EV, best book. |
| `edgeline/data/opendota.py` | 133 | Dota 2 pro history via OpenDota's SQL explorer, paged by match id. |
| `edgeline/data/bo3.py` | 177 | CS2 history from bo3.gg's JSON API (`api.bo3.gg/api/v1`: matches -> games -> `games/{id}/players_stats`), tiers S to C, thread pool per map, newest-first paging to the cutoff, `--refresh-days` re-fetch for late clan-tag fixes. |
| `edgeline/data/vlr.py` | 205 | Valorant history scraped from vlr.gg match pages (per-map player rows; no headshot counts published). |
| `edgeline/data/lolesports.py` | 217 | LoL history from Riot's esports API (leagues -> tournaments -> completed events, merged with the schedule feed) and the livestats window feed (final frame per game); the end-frame lookup is clamped to now. |
| `edgeline/data/breakingpoint.py` | 123 | Call of Duty (and CS2) history from Breaking Point's Supabase REST backend (`player_stats`, `games`, `matches`, `teams`, `maps`, `modes`; anon key from the site bundle, override `EDGELINE_BREAKINGPOINT_KEY`). |
| `edgeline/data/leaguepedia.py`, `oracles_elixir.py` | 114, 75 | Alternative LoL loaders (rate-limited / local CSV); not used by the live models. |
| `edgeline/features/build.py` | 359 | As-of feature engineering (section 5). |
| `edgeline/features/mappool.py` | 97 | Map-pool expectation for CS2 and COD (team's map frequencies over its last 20 maps x player's rolling mean per map). |
| `edgeline/models/props.py` | 397 | Training pipeline per (sport, stat) (section 6). |
| `edgeline/models/distributions.py` | 144 | Negative binomial per map, over/under/push at a line, dispersion MLE, gamma-frailty multi-map sums, `fair_line` (median-fair .5 line). |
| `edgeline/models/copula.py` | 117 | Gaussian copula over (player, team, map) components with rho_self / rho_team / rho_opp; `sum_over_under_push`, `joint_hit_probability`, `joint_hit_pmf` (distribution of legs hit, used for flex ladders). |
| `edgeline/models/predict.py` | 385 | `BoardPricer` (section 7), `NameResolver` (player names by normalized core, team codes by roster majority within 180 days), `market_is_unproven`. |
| `edgeline/models/stacks.py` | 76 | Joint probability of two same-game legs, lift over independence, break-even 2-pick multiplier. |
| `edgeline/models/winner.py` | 87 | Map-winner LightGBM classifier on team as-of features vs an Elo-gap logistic baseline; series probabilities. |
| `edgeline/models/banlist.py` | 62 | One-sided binomial test of a player's graded picks against the model's own probabilities (min 15 picks, alpha 0.05). |
| `edgeline/models/backtest.py` | 263 | Walk-forward backtest vs synthetic setters (naive trailing mean; book-like model), single map and two-map sums. |
| `edgeline/models/replay.py` | 83 | Price already-settled captured lines with models trained only before a cutoff (out of sample on real lines). |
| `edgeline/ev/payouts.py` | 150 | Ladders and contest guarantees (section 8). |
| `edgeline/ev/math.py` | 95 | `leg_ev`, `poisson_binomial_pmf`, `slip_ev_from_pmf`, `slip_hit_prob`, `breakeven_hit_rate`, `kelly_fraction`, `ev_table`. |
| `edgeline/ev/simulate.py` | 78 | Monte Carlo bankroll paths (median / p25 / p75, drawdown). |
| `edgeline/slips/builder.py` | 131 | Greedy EV-optimal slips from bettable legs (section 9). |
| `edgeline/slips/stacks.py` | 115 | Same-match, same-stat, same-direction stacks priced with the copula; two-team rule. |
| `edgeline/slips/best.py` | 153 | The best slip at every price with growth ranking and Kelly stakes. |
| `edgeline/slips/horizon.py` | 32 | Horizon (today + `SLIP_HORIZON_DAYS` in New York) and freshness (`still_listed`: seen within 30 minutes of the book's latest pull). |
| `edgeline/grading/grade.py` | 104 | Settlement (section 10). |
| `edgeline/grading/results.py` | 65 | Tracker-style results (open vs current, leg ROI, 4-pick ROI), standard lines only. |
| `edgeline/grading/roi.py` | 116 | Wilson interval, parlay ROI, sample size needed, match-cluster bootstrap (`match_key`, `cluster_ci`, `MIN_CLUSTERS = 5`). |
| `edgeline/grading/clv.py` | 109 | Closing-line value. |
| `edgeline/grading/track.py` | 100 | Day-on-day market tracker with the pre-stated gate lift rule. |
| `scripts/watch.sh` | | Start or restart the line watcher (`edgeline lines watch --sports lol,cs2,val,dota,cod --interval N`, log `data/logs/watch.log`). |
| `scripts/build_workbook.py` | | The xlsx. |
| `scripts/stack_backtest.py` | | Walk-forward stack backtest (`python scripts/stack_backtest.py cs2 kills 3,4,5 0.55`). |

## 4. Data sources and what is loaded

| Sport | Source | Rows in `player_games` | Games / maps | Dates | Notes |
|---|---|---|---|---|---|
| Dota 2 | OpenDota SQL explorer | 304,600 | 30,460 matches | 2024-09-25 to 2026-09-26 | pro matches with league tier, series ids, notable player names |
| CS2 | bo3.gg API | 331,344 | 33,159 maps | 2025-04-01 to 2026-09-26 | tiers S to C; kills, deaths, assists, headshots, ADR, KAST, first kills/deaths, rating, rounds, map name |
| LoL | Riot esports API + livestats | 103,830 | 10,383 games | 2025-01-13 to 2026-09-26 | every league Riot lists except TFT (38 leagues); LPL thin |
| Valorant | vlr.gg | 86,542 | 8,655 maps | 2025-08-01 to 2026-09-26 | no headshot counts, so Valorant headshots are not priced |
| Call of Duty | Breaking Point | 49,051 | 6,108 maps | 2024-12-06 to 2026-08-09 | 2024-25 and 2025-26 seasons; mode per map (HP, S&D, Overload); board empty until December |

Book data captured: 2,876 lines (PrizePicks cs2 1,765 of which 1,669 standard; dota 226; lol 266 of which 211
standard; val 129; Sleeper cs2 490), 69,176 line snapshots, 2,630 predictions, 1,765 grades, 2,554 Kalshi
quotes, 140 Kalshi grades, 27 saved slips. Capture started 2026-09-25.

History refresh commands: `edgeline history opendota --since <date>`, `edgeline history bo3 --since <date>`
(one request per map; 60 days is about an hour), `edgeline history vlr --pages N --since <date>` (1.5 s per
match), `edgeline history lolesports --since <date>`, `edgeline history breakingpoint --sport cod --since <date>`,
`edgeline history stats`. Full reload dates used on a cold container: opendota 2024-09-25, bo3 2025-04-01, vlr
pages 60 since 2025-07-20, lolesports 2025-01-01, breakingpoint 2024-11-01.

Why these sources: HLTV is Cloudflare-walled, Leaguepedia rate-bans after one burst, Oracle's Elixir hits Drive's
download quota, breakingpoint.gg is a client-rendered app whose bundle ships an anon Supabase key with
anonymous reads allowed.

## 5. Features (`features/build.py`)

Everything is as-of: shift-then-roll, or a rating updated after each game, so a training row never sees its own
outcome. The same aggregations without the shift give the current state at prediction time (`current_state`).

- `p_*` player form: EWM (half-life 8 games) and rolling means over 5, 10, 20 games and 10-game std of kills,
  deaths, assists, headshots; kill share (`p_kshare_mean10`), games played, game length, days since last game,
  per-round rates (`p_<stat>_pr10`, `p_rounds_mean10`) for the round-based shooters, team games in the last 60
  days (`p_team_games60`, recorded as the `tenure60` note).
- `pr_*` player form within a role (role = position in the MOBAs, game mode in COD).
- `t_*` team: kills for and against over 10, win rate, game length, rounds, games, Elo (K = 24, start 1500).
- `o_*` opponent: the same, plus role matchup: what the opponent's same-role player scores (`o_role_kills10`,
  drives deaths) and what the opponent concedes to that role (`o_role_conceded10`, drives kills).
- Strength: `matchup_win_diff`, `elo_diff`, `elo_absdiff`, `p_win_elo` (the missing moneyline's proxy).
- Context: `game_number`, `playoffs`, categoricals `role`, `league`, `tier`.
- `p_map_expected_kills`: map-pool expectation (CS2, COD).
- `exp_rounds` and `<stat>_pr_x_rounds`: expected rounds from both teams' recent maps times the player's rate.
- Off by default, measured and left off: `EDGELINE_ROUND_FEATURES=1` (Valorant 65.4% vs 64.5%, noise),
  `EDGELINE_EXTRA_STATS=1` (bo3.gg ADR/KAST/first kills/rating: CS2 kills 57.0% vs 58.1%), `EDGELINE_KNOWN_MAP=1`
  (post-veto experiment: the model sees the actual map's form).
- COD: features per (player, mode); pricing maps a map number to its mode (1 HP, 2 S&D, 3 Overload, 4 HP, 5 S&D).

## 6. The prop model (`models/props.py`)

Per (sport, stat), 15 models in total: dota, cs2, val, lol, cod x kills, deaths, assists; cs2 headshots.

1. Time-ordered split, last 20% of dates held out.
2. LightGBM Poisson objective for the per-map mean: `learning_rate 0.03, num_leaves 31, min_data_in_leaf 60,
   feature_fraction 0.8, bagging_fraction 0.8, bagging_freq 1, lambda_l2 1.0, seed 7`, early stopping on the
   validation split (`edgeline model tune` sweeps a six-point grid and saves `models/artifacts/<sport>_params.json`).
3. Negative-binomial dispersion `r` by maximum likelihood on held-out (y, mu).
4. Residual correlations from standardized residuals: `rho_self` (same player across maps of a series, turned
   into a shared gamma frailty), `rho_team` (teammates in a map), `rho_opp` (opponents in a map).
5. Drift-aware calibration on the most recent 90 days of the held-out split (at least 500 rows): a mean-bias factor
   mean(actual) / mean(predicted) folded into every projection, the dispersion, and a Platt-style
   `LogitCalibrator` (p_cal = sigmoid(a * logit(p_raw) + b)) fit on NB tail probabilities at lines placed at
   offsets -4 to +4 around the mean, for single maps and two-map sums. An isotonic calibrator was tried first and
   rejected: it clipped everything outside its support to 0 or 1 and produced fake 99.9% legs.
6. Metrics saved per model: MAE vs player last-10 mean vs global mean, Poisson NLL, Brier and log-loss raw and
   calibrated, reliability tables, hit rate at the 60% threshold, naive-line policy hit rates, feature importance.

Live models (validation split; MAE = mean absolute error of the per-map mean):

| Model | Train / valid rows | MAE model / last-10 / global | NB r | rho self / team / opp | Brier (cal.) | Policy >= 60% vs naive line |
|---|---|---|---|---|---|---|
| dota kills | 220,226 / 55,057 | 2.899 / 3.004 / 3.552 | 3.36 | 0.051 / 0.269 / -0.067 | 0.1977 | 66.1% (n 24,346) |
| dota deaths | same | 2.451 / 2.533 / 2.845 | 6.07 | 0.068 / 0.528 / -0.113 | 0.1893 | 66.7% (n 25,303) |
| dota assists | same | 5.507 / 5.748 / 6.024 | 3.68 | 0.064 / 0.744 / -0.056 | 0.2356 | 66.5% (n 23,446) |
| cs2 kills | 253,757 / 63,451 | 4.209 / 4.417 / 4.320 | 13.74 | 0.040 / 0.372 / 0.282 | 0.2205 | 66.6% (n 27,966) |
| cs2 deaths | same | 3.067 / 3.307 / 3.168 | 67.73 | 0.066 / 0.810 / 0.484 | 0.1954 | 69.5% (n 35,460) |
| cs2 assists | same | 1.991 / 2.075 / 2.057 | 12.05 | 0.016 / 0.210 / 0.158 | 0.1605 | 66.4% (n 26,822) |
| cs2 headshots | 253,768 / 63,447 | 2.702 / 2.803 / 2.915 | 13.44 | 0.021 / 0.182 / 0.145 | 0.1892 | 65.9% (n 25,759) |
| lol kills | 76,281 / 19,071 | 1.968 / 2.090 / 2.368 | 3.27 | -0.002 / 0.168 / -0.061 | 0.1688 | 67.6% (n 7,866) |
| lol deaths | same | 1.646 / 1.731 / 1.745 | 10.57 | 0.003 / 0.431 / -0.122 | 0.1527 | 67.4% (n 8,905) |
| lol assists | same | 3.444 / 3.675 / 3.936 | 4.04 | -0.032 / 0.641 / -0.156 | 0.2094 | 67.9% (n 8,300) |
| val kills | 62,584 / 15,651 | 4.116 / 4.300 / 4.274 | 17.25 | 0.024 / 0.243 / 0.157 | 0.2205 | 66.7% (n 6,039) |
| val deaths | same | 2.779 / 2.953 / 2.874 | 200.0 | 0.059 / 0.764 / 0.308 | 0.1922 | 68.6% (n 7,836) |
| val assists | same | 2.408 / 2.477 / 2.691 | 8.61 | 0.012 / 0.143 / 0.075 | 0.1787 | 64.4% (n 5,591) |
| cod kills | 38,005 / 9,502 | 4.104 / 9.537 / 9.205 | 39.82 | 0.074 / 0.332 / 0.286 | 0.2157 | 65.2% (n 3,999) |
| cod deaths | same | 3.193 / 9.334 / 9.037 | 200.0 | 0.106 / 0.761 / 0.479 | 0.1937 | 69.3% (n 4,769) |
| cod assists | same | 2.459 / 4.520 / 4.357 | 27.92 | 0.048 / 0.294 / 0.230 | 0.1844 | 63.5% (n 5,692) |

Reading: in the MOBAs teammates move together and opponents slightly against; in the round-based shooters every
player in the match moves together (more rounds, more of everything), so same-direction stacks are the only ones
worth pricing there. The CS2 model beats the naive baselines by only 2 to 5% MAE.

## 7. Pricing (`models/predict.py`, `BoardPricer`)

1. Resolve the board's names: players by normalized name (tag prefixes and suffixes stripped), team codes by
   majority vote of their players' latest team (rosters from the last 180 days). Unresolved names get a
   `player_unmapped` or `opponent_unmapped` note and no price.
2. One (player, map) component per line: `MAP n` is one component, `MAPS a-b` is b-a+1, combos add a component
   per player. Predict the mean per component with the stat's model (COD: mode by map number).
3. Market prior: each component's mean is shrunk 25% toward the book's implied per-map line
   (`DEFAULT_MARKET_SHRINK`, note `shrink:0.25`). Replay at 0.25, 0.5 and 0.75 gave 52.9%, 51.9%, 52.1% on the same
   378 CS2 leans, so heavier shrink does not help.
4. Price: a single component analytically from the NB (over, under, push at integer lines); multi-component lines
   by Gaussian-copula simulation of the sum.
5. Lean = the side with the higher probability; `prob` is its calibrated probability; EV per leg =
   prob x (odds - 1) - (1 - prob) with pushes refunded, at the book's per-leg odds (PrizePicks 1.7783 = 10^(1/4);
   Sleeper uses each side's own multiplier).
6. Bettable = prob >= 60% and EV >= 5%, and none of: market gated (`market_unproven`), non-standard odds type
   (`odds_type:demon|goblin`), voidable (`voidable`; refined by the series format, note `boN`), line moved more
   than 10% against the lean since open (`bumped_against:+x%`), status not pre-game, banned player.
7. Notes recorded on every prediction: `tenure60:<games>` and `rest:<days>` (the evidence-depth split), plus the
   flags above, so the forward record can be split without reconstruction.
8. Predictions are stored per (book, projection_id, model_version); the results tracker keeps the latest version
   per line.

Voidable: a `MAPS 1-3` line on a best-of-3 is voidable (the series can end 2-0); on a best-of-5 it is not. LoL
formats come from Riot's schedule (every league, 12 hours back to 96 hours ahead, matched by code, name or
time within 90 minutes). Other sports assume best-of-3 unless the loader knows better.

## 8. Payouts and EV (`ev/payouts.py`, `ev/math.py`)

PrizePicks ladders, net per $1 by legs hit (verified against prizepicks.com's payouts page on 2026-09-27):

| Price | Ladder | Top | Break-even leg | EV at a 60% leg | Paid at 60% |
|---|---|---|---|---|---|
| POWER 2 | 3x | 3x | 57.7% | +8.0% | 36% |
| POWER 3 | 6x | 6x | 55.0% | +29.6% | 22% |
| POWER 4 | 10x | 10x | 56.2% | +29.6% | 13% |
| POWER 5 | 20x | 20x | 54.9% | +55.5% | 8% |
| POWER 6 | 37.5x | 37.5x | 54.7% | +75.0% | 5% |
| FLEX 3 | 3x / 1x | 3x | 57.7% | +8.0% | 65% |
| FLEX 4 | 6x / 1.5x | 6x | 55.0% | +29.6% | 48% |
| FLEX 5 | 10x / 2x / 0.4x | 10x | 54.3% | +43.4% | 68% |
| FLEX 6 | 25x / 2x / 0.4x | 25x | 54.2% | +66.4% | 54% |

Contest routing (observed on four of the account's lineups, 2026-09-26/27): a lineup whose picks all come from one
match is routed to a contest card (leaderboard; first place pays the standard multiplier, otherwise a minimum
guarantee). Known guarantees: 3-pick power 3.5x; 3-pick flex 2.75x on 3 of 3 and 0.5x on 2 of 3. Break-even on
those guarantees is 65.9% and 65.6% per leg, so a single-match 3-pick at 60% legs is -24% / -19% EV unless it
wins the contest. Sizes not yet read off the app (2, 4, 5, 6) fall back to the standard ladder and are labelled
"guarantee unverified". Multi-match lineups pay the standard ladder. `EDGELINE_PRIZEPICKS_FORMAT=arena` prices
every lineup on the guarantee.

Underdog ladders (power 2 to 8, flex 3 to 8) and Sleeper (product of per-leg multipliers) are in the same file.
Slip EV is the exact Poisson-binomial over the ladder; `breakeven_hit_rate` bisects it; `ev table` and
`ev simulate` reproduce LCSLarry's calculator.

## 9. Slips

Builder (`slips/builder.py`, `edgeline slips build`): bettable legs only, ranked by leg EV, at most one leg per
player and one per game per slip, skip lines in `used_lines`, players from at least two teams, exact
Poisson-binomial slip EV, pushes treated as losses inside a slip.

Horizon and freshness (`slips/horizon.py`): only games starting before the end of tomorrow in New York
(`--days` widens it), and only lines seen within 30 minutes of the book's latest pull.

Stacks (`slips/stacks.py`, `edgeline slips stacks`): legs from one match, one stat, one direction; k-1 legs from
one team plus the best same-lean leg from the opponent (the two-team rule); priced jointly with the copula
(`joint_hit_pmf`), so flex ladders get credit for partial hits; single-match lineups priced on the contest
guarantee where known.

Best slip at every price (`slips/best.py`, `edgeline slips best --bankroll 2000 --kelly 0.25 --cap 0.005
--shrink 0.05 --days 1`): for every ladder the book sells, the highest-value slip from the bettable board
(independent legs via the builder, stacks via the copula), ranked by growth rate rather than EV. Stake policy:
5 points off every leg's probability, a stack's correlation credit halved, growth-optimal Kelly fraction on the
whole payout distribution (ternary search on E[log(1 + f x net)]), a quarter of that, capped at 0.5% of bankroll
($10 on $2,000) until the bettable record's match-cluster interval clears break-even (then `--cap 0.01`).
`edgeline slips use <ids>` marks legs used after a slip is placed.

Why growth, not EV: a 37.5x ladder that pays 5% of the time can have a higher EV and a lower growth rate than a
10x flex that pays on half its slips. At a 60% leg the 5- and 6-pick flex return +43% and +66% and pay on 68% and
54% of slips; the 4-pick power returns +30% and pays 13%.

## 10. Grading and evidence

Settlement (`grading/grade.py`, `edgeline grade --sport <s> --book <b>`, lines older than 4 hours): find the
player's games within 30 hours of the start, take the series closest to the start, sum the stat over the line's
map range; if fewer maps were played than the range needs and the line is voidable, grade `void`; a partially
loaded series does not void later-map lines until the loaded maps show a decided series (three map wins, or two
after six hours). Results against both the opening and the current (closing) line; pushes on integer lines.
Completeness (added 2026-09-27 after 40 lines were found graded on partial maps): every map in the line's range
must have a full roster (10 players; 8 in COD), a plausible kill total (CS2 and Valorant 60, LoL 8, Dota 12, COD 40)
and a value for the stat, or the line stays pending; CS2 and Valorant "maps" of three rounds or fewer are technical
restarts and are dropped before the series is numbered (`game_completeness`, `drop_junk_maps`); grades of lines
that started within the last three days are recomputed on every pass (`--regrade-days`), and `edgeline grade --all`
regrades a sport from scratch.

Blind same-side rate and selection value (`grading/roi.py`, `blind_rates`): for a slice, the blind rate is the share
of all settled lines that went under (or over), weighted by how often the model leaned each way; the selection
value is the model's hit rate minus that. It is printed by `edgeline roi` and carried in the record feed, and it is
the only figure that separates the model from the market's tilt.

Results (`grading/results.py`, `edgeline results --by sport`, `--bettable-only`): one row per unique standard line
(latest model version), hit rate, leg ROI at the book's per-leg odds, 4-pick parlay ROI = hit^4 x 10 - 1.

ROI gauge (`grading/roi.py`, `edgeline roi`): Wilson 95% interval on the hit rate, parlay ROI at the point and the
bounds, lines needed for the lower bound to clear 56.2% and 60%, and a match-cluster bootstrap (2,000 resamples of
whole matches, match = unordered team pair plus start day; not printed under five matches) because lines from one
match win or lose together.

Closing-line value (`grading/clv.py`, `edgeline clv --by sport|edge`): for every started line, the lean is the
projection against the captured opening line, the close is the last snapshot before the start; "for" when the
close moved toward the lean, "against" when away; exact binomial test at 50%; bins by the projected gap.

Day-on-day tracker (`grading/track.py`, `edgeline track --sport cs2 --stat kills|headshots`): per match day, our
leans' record, matches, under- and over-lean hit rates, the blind under rate (share of all lines settling under),
and selection value = our leans' hit rate minus the blind rate of the same side. The gate lift rule is pre-stated:
trailing ten matches with 150+ lines at 58% or better and selection value +3% or better.

Replay (`models/replay.py`, `edgeline replay`): models trained only before a cutoff price every settled line at its
opening number. Backtest (`models/backtest.py`, `edgeline model backtest [--maps 2]`): monthly walk-forward against
a naive setter (trailing 10-game mean at the median-fair .5) and a book-like setter (a separate LightGBM on the
basic averages a small book would use), live rules applied exactly. Stack backtest: `scripts/stack_backtest.py`.
Ban list (`edgeline banlist update`): one-sided binomial test, min 15 picks, alpha 0.05; empty today.

## 11. The record as of 2026-09-27 15:00 UTC (standard lines, opening line, after the grading correction)

| Slice | Record | Hit rate (95% Wilson) | Matches | Match-cluster 95% | Blind same-side rate | Selection value |
|---|---|---|---|---|---|---|
| All model leans | 581-473 | 55.1% (52.1 to 58.1) | 71 | 50.8 to 59.1 | 52.6% | +2.6 |
| Bettable (>= 60% prob, >= 5% EV) | 94-60 | 61.0% (53.2 to 68.4) | 51 | 50.3 to 71.9 | 58.4% | +2.6 |
| CS2 leans | 417-363 | 53.5% (50.0 to 56.9) | 42 | 48.7 to 58.0 | 51.8% | +1.7 |
| CS2 bettable | 46-46 | 50.0% | | | 53.1% | -3.1 |
| Dota leans | 55-38 | 59.1% (49.0 to 68.6) | 9 | 45.6 to 70.2 | 51.5% | +7.7 |
| LoL leans | 84-54 | 60.9% (52.5 to 68.6) | 17 | 50.0 to 69.2 | 61.2% | -0.4 |
| Valorant leans | 25-18 | 58.1% (43.3 to 71.6) | 3 | too few | 49.3% | +8.9 |

What it shows: the book's esports lines sat above the actual counts on these three days (every under went under 56%
of the time over all lines, 62% in Dota and LoL), so a blind under bettor would have posted most of this record. The
model adds 2.6 points over blind on all leans and on bettable picks, nothing in LoL, and is 3 points worse than blind
on CS2 bettable picks. Whether the market tilt persists is unknown; the model's own contribution is small and
unproven. Earlier versions of this record compared hit rates with 56.2% and not with the blind rate, which overstated
the model. The grading correction of 2026-09-27 (section 10) moved all leans from 613-488 to 581-473 and bettable
from 96-61 to 94-60; 56 CS2 lines wait on bo3.gg completing their maps.

Bettable against the closing line: 90-52 (63.4%). At the current rate about 249 settled bettable lines put the
Wilson lower bound above break-even and about 1,348 above 60%. No slice's match-cluster lower bound clears 56.2%.

Bettable by market (from the README's 01:00 pass): CS2 42-41 (headshots 26-18, kills 16-23), Dota 9-3, LoL 37-9,
Valorant 1-0. Two LoL best-of-fives carry 82 of LoL's lines (FlyQuest vs Shopify Rebellion 36-14, LYON vs Shopify
Rebellion 25-7), which is why the cluster interval matters more than the per-line one.

Closing-line value on 1,596 started lines: 105 moved toward the model, 42 against (71%, p < 0.001; CS2 87-30,
Dota 4-1, LoL 13-9); on bettable picks 10 for, 12 against (45%, p 0.83). By projected gap at the 07:50 read: 0.5 to
1.0 from the open 12-1 for us, 1.0 to 2.0 16-3, over 2.0 3-3. The market confirms the model's moderate
disagreements and not its largest ones, which is where most bettable picks live.

Whose number is better (MAE against the actual count):

| Market | Lines | Model MAE | Book MAE |
|---|---|---|---|
| CS2 kills, live models | 215 | 4.77 | 4.52 |
| CS2 headshots, live models | 168 | 3.33 | 3.27 |
| CS2 replay | 378 | 4.55 | 4.59 |
| Dota kills, live models | 47 | 4.27 | 4.47 |
| Dota replay | 56 | 4.21 | 4.35 |
| LoL replay | 23 | 3.00 | 3.22 |

Replay (models trained before 2026-09-25, every settled line at its open): Dota 39-17 (69.6%), LoL 18-5 (78.3%),
CS2 200-178 (52.9%; headshots leans 57.7%, kills leans 49.0%).

CS2 day-on-day (`edgeline track`):

| Market | 09-25 | 09-26 | Trailing 10 matches | Selection value | Lift rule |
|---|---|---|---|---|---|
| kills (gated) | 70-91 (43.5%) | 114-91 (55.6%) | 70-53 (56.9%, CI 48.1 to 65.3) | +2.9% | not met |
| headshots (open) | 84-67 (55.6%) | 111-82 (57.5%) | 58-51 (53.2%) | +0.4% | not met |

Evidence-depth split on settled CS2 lines (hypothesis, post hoc): players with 11 to 20 games for their team in
the previous 60 days 17-35; lines priced 11 to 30 days after the player's last loaded game 14-24; 21+ games 166-138;
played within three days 132-115.

Other CS2 pockets checked and closed: map-3 lines carry no 1-1 conditioning bias; sum lines sit 0.9 above map-1
plus map-2 as the skew implies; roster changes flagged by bo3.gg do not hurt (29-16); winner-model lopsidedness
does not separate kills leans; favourites' unders 112-68 (62.2%) is the one kills tilt worth a forward test
(unders 79% in maps of 18 rounds or fewer, 40% in overtime maps). A "share" model (our kill shares on the book's
team total) did not beat either number. CS2 progress needs new information (map veto, stand-ins, cross-region
strength), not model tweaks.

## 12. Backtests (edge over synthetic setters, not realized ROI)

Walk-forward, kills, 2026-05 to 2026-09, >= 60% threshold, vs the book-like setter (picks, share of games, hit
rate, 4-pick ROI): LoL 1,690 (5.1%) 63.6% (61.3 to 65.9) +64%; Dota 1,045 (5.9%) 66.4% +95%; Valorant 541 (1.8%)
65.6% +85% (seven months of history; 61.2% on 273 with fourteen); CS2 kills 2,523 (2.7%) 64.2% (62.4 to 66.1) +70%
(eighteen months; 58.5% with six); CS2 headshots 3,395 (3.7%) 64.7% +76%; COD 226 (1.7%) 68.1% (61.8 to 73.9) +116%.
Against the naive setter every sport is 66% to 69%. Two-map sums are at or above single-map rates everywhere
(Dota 68.1%, Valorant 64.3%, LoL 64.7%, CS2 kills 64.6%, CS2 headshots 69.0%). Drift-aware calibration balanced
the sides (share of overs among picks: LoL 16%, Dota 24%, Valorant 63%, CS2 35%, COD 48%).

Stacks (same-team, unders, realized vs independent product, power ROI): CS2 kills 3-leg 30.2% vs 19.9% (+81%),
4-leg 23.9% vs 11.9% (+139%), 5-leg 20.4% vs 7.2% (+307%), copula within two points; LoL 3-leg 30.8% vs 20.7%
(+85%), 4-leg 25.0% vs 12.4% (+150%), 5-leg 24.4% vs 7.3% (+388%), copula under-prices LoL same-team (19.9% vs
25.0%); Dota 3-leg 37.7% vs 20.6% (+126%), 4-leg 32.4% vs 12.4% (+224%), 5-leg 29.8% vs 7.6% (+496%), overs 43.0%
vs 20.7%. Correlation multiplies whatever edge the legs have: the same CS2 stacks on 2026-09-25's real lines went
1-13 because the legs hit 50%. Rule-compliant stacks (one opponent leg) return less than pure same-team ones.

## 13. Markets: what is bettable and why

- Dota and LoL: projections beat the book's number on real lines; positive real-line records; the markets to
  size first (Dota same-team under stacks of three or four legs at 55%+ per leg is the product the backtest
  ranks highest). Sunday LoL playoff series are best-of-5, so `MAPS 1-3` lines there cannot void.
- CS2 headshots: open. 133-92 (59.1%) at gating time, UNDER leans 65% against a 58% blind under rate, replay 57.7%;
  the trailing window has cooled (section 11). Roughly half the CS2 board.
- CS2 kills: gated (`UNPROVEN_MARKETS`) since 16:00 UTC on 2026-09-26: 136-141 over 29 matches, the model's side
  adds nothing over the market's own under rate, and the book's number is more accurate. Priced and graded every
  pass; lifts only on the tracker's rule.
- Valorant: 18-13 over two matches; too thin to say anything.
- COD: backtests at 68.1% on 226 picks; board empty until the CDL season restarts in December.
- Demon and goblin lines: excluded everywhere (MORE-only, own ladders).

## 14. Books and the exchange

| Book | Status |
|---|---|
| PrizePicks | integrated: partner API, opening-line capture, grading, slips, deep links; 429 bursts handled by 12 s pacing |
| Sleeper Picks | integrated: public JSON, CS kills and headshots maps 1-2, per-side multipliers; record 50-42 leans on the first day |
| Underdog | needs the web app's client headers from a browser session (`EDGELINE_UNDERDOG_HEADERS`) |
| Dabble | readable (Valorant, LoL, Dota per-map kills at decimal prices); Australia-only wagering; a second line reference |
| Kalshi | readable; 462 open esports markets with zero volume; orders need the RSA key paired with the API key id in `EDGELINE_KALSHI_KEY_ID` (environment only, never in the repo) |
| ParlayPlay, ThunderPick, Stake, Betr, Chalkboard | blocked from the cloud container; should work from a residential machine |
| Boom Fantasy | needs a logged-in JWT |
| DraftKings Pick6 | no public board found |

Winner model (`models/winner.py`, `edgeline model winner-backtest --sport <s>`): LightGBM binary classifier on
`t_elo, o_elo, elo_diff, t_win10, o_win10, matchup_win_diff, t_kills_mean10, o_kills_mean10, t_oppkills_mean10,
o_conceded_mean10, t_games, o_games, rest_days, h2h_wins, h2h_games, game_number` (`num_leaves 15,
min_data_in_leaf 200, lambda_l2 5`), walk-forward vs an Elo-gap logistic: CS2 8,583 maps log-loss 0.660 vs 0.664,
Valorant 0.673 vs 0.675, LoL 0.624 vs 0.627, Dota 0.637 vs 0.645; accuracy 58% to 64%, below a sportsbook's.

Kalshi results (`edgeline kalshi scan --sports cs2,val,lol,dota --limit 15`, `edgeline kalshi grade`): 140 graded
markets, 125 with two-sided quotes (108 CS2, 7 Dota, 6 LoL, 4 Valorant). Kalshi's mid is sharper: log-loss 0.603
vs the model's 0.634 overall (CS2 0.614 vs 0.643); the buy-the-edge rule lost 33 cents per dollar over 43 trades at
a 3-cent threshold (CS2 -47% over 31). Median spread 14 cents. Conclusion: nothing to trade until liquidity
arrives (Worlds, CS2 majors) and the model would still need to beat the exchange by more than a book's margin.

## 15. Operations

Database (`data/edgeline.db`, SQLite; `edgeline init` creates it):

| Table | Columns |
|---|---|
| line_snapshots | id, book, projection_id, fetched_at, line, odds_type, status, updated_at |
| lines | book, projection_id, sport, league, player_id, player_name, team, opponent, position, game_id, stat_type, stat, map_from, map_to, combo, combo_players, voidable, board_time, start_time, open_line, open_seen_at, current_line, current_odds_type, last_seen_at, status, odds_over, odds_under |
| player_games | sport, source, game_id, series_id, game_number, date, league, tier, patch, player_name, player_id, team, opponent, role, side, champion (map name in CS2/COD), kills, deaths, assists, headshots, team_kills, opp_kills, game_length, win, playoffs, rounds, adr, kast, first_kills, first_deaths, rating |
| predictions | book, projection_id, model_version, computed_at, line, projection, p_over, p_under, ev_over, ev_under, lean, prob, ev, bettable, notes |
| grades | book, projection_id, graded_at, actual, maps_played, result_open, result_current, source |
| slips | id, created_at, book, slip_type, size, ev, hit_prob, legs_json, link |
| used_lines | book, projection_id, used_at |
| banned_players | sport, player_name, n, hits, expected, pvalue, banned_at |
| alerts_sent | book, projection_id, sent_at |
| kalshi_quotes | ticker, fetched_at, event_ticker, sport, team, opponent, map_index, market_kind, close_time, yes_bid, yes_ask, volume, our_p |
| kalshi_grades | ticker, sport, team, opponent, map_index, event_time, actual, last_fetched_at, yes_bid, yes_ask, our_p, graded_at |

Every CLI command: `init`, `lines pull|watch|show|shop|promos|underdog-dump`, `history opendota|bo3|vlr|lolesports|
breakingpoint|leaguepedia|oracles-elixir|stats`, `model train|tune|metrics|backtest|winner-backtest`, `predict`,
`replay`, `stacks`, `slips build|stacks|best|use`, `ev table|simulate`, `banlist update|show`, `alerts send`,
`grade`, `results`, `roi`, `clv`, `track`, `kalshi scan|grade`, `web`, `export`, `import`.

Watcher: `bash scripts/watch.sh 180` polls all five boards every 180 s, prices new lines as they post, appends
snapshots, and logs to `data/logs/watch.log`. Check it with `pgrep -f 'edgeline lines watc[h]'` (that pattern
does not match the caller's own shell). The cloud container is suspended when the session idles and the outbound
proxy port changes on restore, so the watcher dies or reports "Unable to connect to proxy" and must be restarted
by PID; on a machine that stays up none of this applies.

Routines (Claude Code Remote, bound to the original session):

- Every 3 hours at :59 (`trig_01KTEavyH12iSyng4AYUdaZ6`): cold-container recovery if the database or models are
  missing (init, import, reload histories, train, start watcher); refresh histories since yesterday; grade every
  sport on both books; `roi`, `results`, `clv`, `track` (kills and headshots), `slips best --bankroll 2000`,
  `kalshi scan` and `kalshi grade`; watcher health; `export`, `web --bankroll 2000`, workbook; commit and push;
  report, with a push notification only on a milestone (a cluster bound clearing break-even at 10+ matches, a
  slice turning negative, the CS2 kills lift rule met, the account's own slip legs settling).
- Hourly at :29 (`trig_01Ddn2Q5zdbswt3RiFYo1S5X`): restart the watcher if it died, wait for one completed pull.

Cold start on a new machine: `edgeline init`, `edgeline import`, load histories (section 4 dates), `edgeline model
train --sport <s>` for each sport, `bash scripts/watch.sh 180`.

Environment variables: `EDGELINE_ROOT`, `EDGELINE_DATA_DIR`, `EDGELINE_DB`, `EDGELINE_ARTIFACTS`,
`EDGELINE_USER_AGENT`, `EDGELINE_DISCORD_WEBHOOK`, `EDGELINE_UNDERDOG_HEADERS`, `EDGELINE_BREAKINGPOINT_KEY`,
`EDGELINE_KALSHI_KEY_ID`, `EDGELINE_PRIZEPICKS_FORMAT` (standard | arena), `EDGELINE_ROUND_FEATURES`,
`EDGELINE_EXTRA_STATS`, `EDGELINE_KNOWN_MAP`.

## 16. Website feeds (`edgeline web --bankroll 2000 --days 1`)

`esports-ev.json`: `generatedAt, demo (false), book, horizonEnd, count, opportunities[]` with, per opportunity:
`id` (`prizepicks:<projection_id>`), `bookKey`, `sportKey` (esports_cs2, esports_lol, esports_dota2,
esports_valorant, esports_cod), `eventId`, `commenceTime` (UTC), `homeTeam` (the player's team), `awayTeam`
(opponent), `market` (snake case, e.g. `maps_1_2_kills`), `marketLabel` (the book's text), `participant`,
`outcome` (Over | Under), `point` (current line), `openPoint`, `decimalOdds` (1.7783), `fairProb` (calibrated
probability of the lean), `ev` (per leg), `projection`, `tier` (bettable | lean | gated | voidable), `inHorizon`,
`voidable`, `gated`, `combo`, `link` (PrizePicks deep link), `linkKind` (deeplink), `derivedFrom` (model version),
`quoteAgeMs`. Sorted bettable first, then by probability. Last run: 313 opportunities, 27 bettable, 231 in horizon.

`esports-slips.json`: `generatedAt, book, bankroll, horizonEnd, count, slips[]` with `price` ("POWER 4"),
`slipType`, `size`, `topPayout`, `kind` (independent | same-match stack, with "contest guarantee" or "contest,
guarantee unverified" when single-match), `ev`, `pTop`, `pPaid`, `growth`, `kelly` (full Kelly fraction), `stake`
(quarter Kelly, capped), `legs[]` (`projection_id, player, team, opponent, sport, stat_type, line, lean, prob,
start_time`), `link`, `label`.

`esports-record.json`: `generatedAt, book, breakEvenLeg (0.5623), targetLeg (0.6), slices[]` with `slice, n, wins,
losses, hitRate, ciLow, ciHigh, matches, clusterCiLow, clusterCiHigh` (null under five matches), `blindRate` and
`selectionValue` (hit rate minus blind rate; the model's contribution) for all leans, bettable, and per sport leans
and bettable.

Page: `web/esports.html` (site shell, `/sharpline.css` tokens, `/theme.js`, `/nav.js`); feed base from `?feed=`,
then `window.ESPORTS_FEED_BASE`, then the branch on GitHub; refreshes every five minutes; flags feeds older than
four hours. `web/README.md` has the Vercel rewrite that serves the feeds same-origin under `/v1/esports-*.json`.
Not done on purpose: writing into oddsfloor's Supabase project (only `public.user_state` exists there).

## 17. The account's live entries (slips table)

| Id | Placed (UTC) | Slip | Legs (all standard lines) | Status |
|---|---|---|---|---|
| 24 | 09-27 00:11 | CS2 POWER 4 | ADK (Rooster) MAPS 1-2 headshots OVER 10.5 (61.7%); beastik (SINNERS) OVER 14.5 (60.6%); FL4MUS (GamerLegion) UNDER 20 (64.5%); tENZY (magic) UNDER 20.5 (61.8%) | settles Sun 05:00 to 13:00 ET |
| 25 | 09-27 00:21 | CS2 POWER 4 | same with beastik UNDER 14.5 (39.4%; placed twice by mistake, the second with the wrong side) | same |
| 26 | 09-27 01:45 | LoL POWER 4 ("slip 1") | Zicssi (SLY) MAPS 1-3 kills UNDER 14.5 (67.9%); Aetinoth (SLY) UNDER 17.5 (67.3%); Revenge (DNS.C) UNDER 11.5 (65.1%); Zest (LOS) UNDER 9.5 (66.4%) | settles by 14:00 ET |
| 27 | 09-27 01:50 | LoL POWER 4 ("slip 2", placed as a contest slip) | Loki (C9) UNDER 10.5 (62.1%); Yeon (TL) UNDER 13.5 (62.9%); Kryze (SLY) UNDER 9.5 (60.2%); Feisty (LOS) UNDER 10.5 (61.4%) | settles by 18:00 ET; open question whether the placed version had Kryze or Duduhh |

The routine grades these legs on each pass (join `slips.legs_json` projection ids to `grades`) and sends a push
notification as they settle. Their projection ids are in `used_lines` so later slips do not reuse them.

## 18. Open items

1. Read the 4-pick (and 2, 5, 6) contest "To Win" guarantees off the app so `CONTEST_GUARANTEES` is complete;
   until then single-match 4-picks are labelled unverified.
2. Underdog: capture the client headers from a browser session.
3. Features the book has and the model lacks: match moneyline, drafts and heroes, map vetoes and stand-ins
   (the CS2 gap), cross-region strength.
4. Cross-stat correlations (kills vs deaths) for stacks; the copula under-prices LoL and Dota same-team joints.
5. The CS2 favourites'-unders tilt and the evidence-depth split (`tenure60`, `rest`) are forward tests, not rules.
6. Earlier feature decisions (map pool, two years of Dota, rejected round features) were measured under the old,
   biased setter and not re-measured under the corrected one.
7. Kalshi: re-check when Worlds and the CS2 majors bring liquidity; orders are not implemented.
8. The sample: about 440 settled bettable lines for the interval to clear break-even at the current rate.
9. bo3.gg partial maps: 56 CS2 lines are pending until the source completes their maps; a map that never completes
   leaves its lines pending forever (Breaking Point's CS2 table is a possible second source for those maps).
10. The market tilt: Dota and LoL lines went under 62% of the time over three days. Find out whether it persists
    (it is what the LoL record is made of) and whether the model can add to it; on CS2 bettable picks it subtracts.

## 19. Conventions for whoever continues

- Report real-line records with intervals, the match-cluster bound, the blind same-side rate and the selection
  value; never a backtest as ROI, and never a hit rate without the blind rate next to it.
- Do not tune thresholds, shrink, or ladders toward a target; every change needs a verifiable reason.
- Standard lines only; two-team rule; horizon and freshness on every proposal; contest guarantee on single-match.
- Keep `data/exports` and `data/web/v1` committed; history tables and models are re-creatable, the line record is not.
- Commits carry the session's co-author trailer; no model identifiers in repository artifacts.
