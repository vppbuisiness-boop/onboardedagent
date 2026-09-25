# edgeline

An open, self-hosted version of the LCSLarry-style esports player-prop model: capture
pick'em lines the minute they post, price every prop with a calibrated per-map count
model, compute expected value against the book's payout ladder, auto-build slips, and
grade every prediction publicly against both the opening and closing line.

The research that motivated the design is in `research/lcslarry-esports-model-report.md`.

## What is built (v0.1)

| Layer | Status | Notes |
|---|---|---|
| Line capture: PrizePicks | working | `partner-api.prizepicks.com` (no bot wall). First sighting = opening line; every poll appends a snapshot. LoL, CS2, VAL, Dota2, COD boards. |
| Line capture: Underdog | stub | Endpoint needs the web app's client headers; set `EDGELINE_UNDERDOG_HEADERS`. Unverified. |
| History: Dota 2 | working | OpenDota SQL explorer, professional tier, per-player per-match rows with series ids. |
| History: LoL | written, blocked here | Leaguepedia Cargo loader (paced) and Oracle's Elixir CSV loader. Both sources throttled the research container; run them from your own machine. |
| History: CS2, VAL, COD | not yet | HLTV is Cloudflare-walled; vlr.gg is reachable (scraper to do); COD via Breaking Point. |
| Features | working | Strictly as-of rolling player / team / opponent stats (shift-then-roll), no leakage. |
| Model | working | LightGBM Poisson mean per map, negative-binomial dispersion by MLE, within-series correlation via shared gamma frailty, isotonic calibration. |
| Pricing | working | P(over/under/push) per line, multi-map sums by Monte Carlo, combos, EV at the book's implied leg odds, bettable rules (min prob, min EV, no demon/goblin, skip voidable, skip lines bumped against you). |
| Slip builder | working | Greedy EV-optimal, one leg per game/player, exact Poisson-binomial slip EV on the payout ladder, PrizePicks deep links, used-line tracking. |
| Grading and results | working | Settles lines from history rows, open vs current line, leg ROI and 4-pick parlay ROI in the same convention as the public tracker. |
| EV tools | working | Payout ladders (PrizePicks, Underdog), break-even rates, Monte Carlo bankroll simulator. |

## Quick start

```bash
pip install -e .            # Python 3.10+
edgeline init
edgeline lines pull --sports lol,cs2,val,dota,cod   # snapshot the boards (paced; ~1 min)
edgeline history opendota --since 2025-09-25        # ~25 s, ~90k pro player-match rows
edgeline model train --sport dota                   # ~30 s for kills, deaths, assists
edgeline predict --sport dota --show-all            # price the board, star bettable lines
edgeline slips build --sports dota --size 3         # EV-optimal 3-pick powers with tail links
# later, once matches finish:
edgeline history opendota --since <yesterday>       # refresh results
edgeline grade --sport dota
edgeline results --by sport
```

Keep the opening-line capture running (the edge is at open):

```bash
edgeline lines watch --sports lol,cs2,val,dota,cod --interval 90
# or cron: */2 * * * * cd /path/to/repo && edgeline lines pull >> data/pull.log 2>&1
```

## First real run (2026-09-25)

Dota 2, one year of OpenDota history (92,710 player-match rows, 9,271 matches), time-split
validation on the last 20% of dates (from 2026-05-16):

| Stat | MAE model | MAE player mean (last 10) | MAE global mean | NB r | series corr | Brier raw -> calibrated |
|---|---|---|---|---|---|---|
| kills | 2.945 | 3.098 | 3.499 | 3.26 | 0.087 | 0.2414 -> 0.2383 |
| deaths | 2.567 | 2.647 | 2.881 | 5.46 | 0.115 | 0.2445 -> 0.2383 |
| assists | 5.684 | 5.939 | 6.100 | 3.61 | 0.132 | 0.2521 -> 0.2475 |

Calibration after isotonic fitting is flat across buckets (e.g. kills: predicted 0.63 vs
observed 0.63 in the 0.6-0.7 bucket, n=6,309). On the live PrizePicks Dota board that day,
37 of 40 lines priced and 8 cleared the 60% / +5% EV defaults.

Read the validation numbers honestly:

- The "policy >= 60%" hit rates printed by `model train` (66% kills, 65% deaths) are against
  synthetic lines placed around the model's own projection, not against book lines. Books
  are sharper than that, so the realized hit rate on real lines will be lower. The only
  number that matters is the graded hit rate on real opening lines, which accrues as you
  run `lines pull` and `grade` over time.
- The model beats the naive "player's last-10 average" by about 5% MAE. LCSLarry's edge
  presumably comes from richer inputs: match moneyline odds (win probability drives kills
  in MOBAs), draft/hero data, map picks in CS2/COD, and roster-change handling. Those are the
  next features to add.
- Pushes on integer lines refund the stake; the pricer accounts for that. Inside a slip the
  builder treats a push as a loss (conservative).
- `MAPS 1-3` props are flagged voidable (assumed best-of-3) and skipped by default, like the
  original product. Verify PrizePicks' current void rules for your jurisdiction.

## How pricing works

1. `features.build` turns `player_games` into as-of rows: EWM and rolling means/stds of the
   player's kills/deaths/assists, kill share, game length, days since last game; team kills,
   kills conceded, win rate; the same for the opponent; map number; playoffs; role; league.
2. `models.props.train` fits a LightGBM Poisson model for the per-map mean, then fits the NB
   dispersion `r` on held-out games, estimates the correlation of standardized residuals
   between maps 1 and 2 of the same series, converts it to a shared-frailty variance `phi`,
   and fits isotonic calibration on tail probabilities at lines near the mean.
3. `models.predict.price_board` resolves book names to history names (players by
   normalized name; team codes by majority vote of their players' latest team), builds one
   feature row per map per player, and computes P(over), P(under), P(push). Multi-map and
   combo lines are Monte Carlo sums with shared frailty (`models.distributions`).
4. EV per leg uses the book's implied per-leg odds from the 4-pick power ladder
   (PrizePicks 10x -> 1.778 per leg, break-even 56.2%), with pushes as refunds.
5. `slips.builder` ranks bettable legs by EV, enforces exposure limits, and prices each slip
   with the exact Poisson-binomial over the ladder (`ev.math`).

## Layout

```
edgeline/
  books/       prizepicks.py (client + open/current tracking), underdog.py (stub), normalize.py
  data/        opendota.py, leaguepedia.py, oracles_elixir.py  -> player_games
  features/    build.py       as-of feature engineering
  models/      distributions.py (NB, frailty, MC sums), props.py (train/calibrate), predict.py
  ev/          payouts.py (ladders), math.py (leg/slip EV, break-even, Kelly), simulate.py
  slips/       builder.py
  grading/     grade.py (settlement), results.py (tracker-style summary)
  db.py, config.py, cli.py
tests/         23 tests (EV math, normalization, distributions, leakage, builder, grading)
```

## Roadmap

1. LoL history from your own machine (`edgeline history leaguepedia --since 2025-01-01` or drop
   Oracle's Elixir CSVs in `data/raw/` and run `edgeline history oracles-elixir`), then
   `model train --sport lol`.
2. Valorant scraper (vlr.gg is reachable), then CS2 (HLTV needs a browser), COD (Breaking Point).
3. Add match moneyline odds as a feature (OddsPapi / The Odds API / Polymarket).
4. Correlated stacks: estimate joint hit rates for same-game pairs and compare with the book's
   reduced payout, as the original product does.
5. Underdog headers, then ParlayPlay/Dabble/Sleeper.
6. Discord webhook alerts on new bettable lines from `lines watch`.

## Disclaimer

Research software. Gambling involves risk; nothing here guarantees profit. Scraping book
endpoints may violate their terms of service. Check the laws in your jurisdiction.
