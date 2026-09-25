# LCSLarry Esports Betting Model: Research Report and Replication Blueprint

Compiled 2026-09-25 from lcslarry.com (home, about, FAQ, onboarding, terms, 15 blog posts), the site's shipped JavaScript bundles (in-app tutorials, help center, settings keys, results-page methodology), the standalone EV calculator at ev.lcslarry.com (payout tables and simulator source), the @LCSLarry X account (tweets pulled via Twitter's syndication API plus attached images), and the Whop storefront.

Everything in Part 1 through Part 4 is sourced. Part 5 (replication) mixes verified facts with clearly labeled inference. Larry has never published the model internals ("I can't spill the secret sauce"), so the exact algorithm is unknown; what follows is the most complete picture the public surface allows.

---

## 1. What it is, in one paragraph

LCSLarry is a subscription SaaS ($99.99 per 2 weeks, $149 per month, $1,499 per year, 3-day free trial, sold through Whop and bundled with the "Juiced Bets VIP" Discord) that runs a fully automated esports player-prop pricing engine. It scrapes every esports player prop from 13 pick'em apps and sportsbooks (PrizePicks, Underdog, Chalkboard, ParlayPlay, Dabble, Sleeper, Boom, Betr, Pick6, Thunderpick, Bovada, Rebet, Stake) about once a minute, produces its own projection and hit probability for each line across five games (League of Legends, CS2, Dota 2, Valorant, Call of Duty), computes expected value against each book's payout, flags lines that clear thresholds (default: at least 60% hit probability and at least 5% EV), auto-builds EV-optimal parlays with correlation handling, pushes Discord alerts when new lines drop, and publicly grades every prediction. The core thesis is that esports player props are a thin, under-modeled market where opening lines are frequently mispriced and a calibrated player-level model beats the books, especially at open.

Builder: "Larry" (@LCSLarry), self-described software engineer and data scientist building esports models since 2023, claims six figures of personal profit before being limited by every book, launched the product publicly in April 2025 (first tweet: "43% ROI over 2,000 tracked bets").

---

## 2. Verified track record claims

All figures below are Larry's own, from the public results tracker. They are theoretical: "Theoretical results based on graded LCSLarry predictions. Not actual placed bets" (footer on his shared graphics). Measurement conventions (from the results page source code):

- Default view is opening lines, 4-pick POWER, all books, at least 60% probability, all esports, $100 units.
- Per-bet P&L uses each source's implied per-leg odds from a 4-pick POWER: about 1.78 decimal on PrizePicks (10x) and 1.86 on Underdog (12x). Selection payout multipliers (Underdog's per-pick multipliers) are applied.
- Total P&L assumes 4-leg parlays: "For every 4 bets, we place a 1-unit parlay," so parlay ROI compounds the per-leg edge.
- "All Books results are deduplicated across books by unique line. Each prediction contributes 1.0 total across its unique lines."
- Sportsbook (Thunderpick, Stake, Bovada) P&L is straight bets at actual odds, 1 unit each.

| Period | Bets | Win rate | ROI | Notes |
|---|---|---|---|---|
| 2025 (Mar to Dec) | 23,281 | 60.1% | +30.3% (+1,761 u) | 9 of 10 months green; Dec was -11.1% |
| Jan 2026 | 4,051 | 58.6% | +16% | Valorant added Jan 8 (+62% ROI first 3 weeks) |
| Feb 2026 | 4,339 | 60.5% | +31.5% | New CS2 (Feb 13), new Dota, new COD (Feb 19) |
| Mar 2026 | 4,003 | ~59.5% | +25.7% | Thunderpick straight betting live |
| Apr 2026 | 3,634 | 60.5% | +32.4% | New LoL model (Apr 15), player ban list |
| May 2026 | 4,685 | 59.6% | +23.3% | Rebet, Chalkboard live; Dota first red sport-month |
| Aug 2026 (to Aug 15) | 1,589 | 59.8% | +34.4% | From tweet graphic, 950-639 |
| Lifetime (site, Sep 2026) | 48,000+ | ~60% | ~29% all-time | "Profitable 15 of 16 months" |
| Opening-line only (Jul 2025 post) | thousands | 63% | +56% | vs closing-line: 61%, +22% |
| Thunderpick straight bets (to Mar 2026) | 11,995 | 57.5% | +4.5% | +10.5% ROI when filtered to EV of at least 10% |

Sanity check on the parlay math: a 60% per-leg hit rate on a 10x 4-pick power yields 0.6^4 x 10 - 1 = 29.6% ROI, which is exactly the "~29% historical ROI" the site quotes. So the headline ROI is a direct function of a 60% leg hit rate, and the real per-leg edge is 6 to 7 points over the 56.2% break-even (Larry states this himself: "Per leg, that's roughly a 6-7% edge").

Caveats to weigh:
- Results are graded predictions at the opening line, not fills. Larry's own blog documents that lines bump within minutes of alerts and that once a line moves more than 10% the edge is mostly gone. Realized member ROI will be lower than the tracker.
- Dedup and "one parlay per 4 bets" are simplifications that ignore correlation between same-match legs on a real slip.
- Losing periods (Jun/Jul 2025, Dec 2025, Dota May 2026, COD Jun 2026) are disclosed, which is a point in favor of the tracker's honesty.

---

## 3. How the system works (verified architecture)

### 3.1 Pipeline

1. **Line ingestion.** Scrapers pull every esports prop from 13 books. Tutorial text: "Our data updates about once a minute." Each line is stored with its opening value and current value; tooltips show "the opening line, the current line, and the same prop's line at every other book." A Discrepancies page lists every prop across all books side by side (raw or percent difference).
2. **Projection.** A per-sport, per-map model produces a numeric projection for each player-prop (e.g. Peyz Maps 1-3 Kills: book 19.5, model 16.5). The FAQ says inputs are "hundreds of thousands of data points about a player's historical performance, the current matchup, and their team's odds of winning the game." The opening/closing blog adds: "sees all of Faker's historical games, the enemy team's entire historical performance, and even factors in the ML odds of which team is likely to win."
3. **Probability scaling (calibration layer).** Explicitly described in the July 2026 update: "a model is really two parts: the projection itself, and the probability scaling that decides how confident to be when that projection disagrees with the book's line. That confidence is what makes a prediction bettable." Calibration is tuned separately for OVER and UNDER ("OVERs were overconfident and UNDERs weren't confident enough... tuned OVER confidence down and UNDER confidence up"), separately per map (COD Map 1, 2, 3 "recalibrated individually"), and with a more conservative scaling for combo lines. Larry's stated calibration standard: "If my model says something hits 60% of the time, that bucket needs to actually hit around 60%."
4. **EV computation.** Per line: hit probability vs the book's implied per-leg odds (DFS: 4-pick POWER payout; variable-payout DFS like Sleeper/Boom and sportsbooks: actual odds). Per slip: full binomial expansion over the payout ladder (Section 3.3).
5. **Bettable filter.** A line gets a green star when it clears all bars: minimum win probability, minimum EV, player exposure limit, game exposure limit. Shipped defaults: "at least 60% probability and at least 5% EV for DFS books." Separate preset for sportsbooks. Per-sport overrides. Advanced filters on by default: skip voidable late-map props, skip lines that moved against the pick, allow correlated legs to pair when a stack improves EV.
6. **Slip builder.** Constructs "EV-optimal parlays" per book with configurable size and type (power/flex), with one-click tail links (PrizePicks deep link format `app.prizepicks.com/?projections=ID-u-LINE,ID-o-LINE`). Placing marks legs as "used" so tomorrow's slips don't repeat them. The "Blender" hand-builds custom slips priced the same way.
7. **Correlation engine.** Auto-generates same-game stacks per book, sortable by stack probability and EV. "Stack EV pits the true joint probability against the book's nerfed payout." Two strategies: EV Optimal vs Prefer Corro.
8. **Alerts.** Discord line-drop alerts per book; notifications when odds move into profitable ranges.
9. **Grading and tracking.** Every prediction graded; results page supports opening vs closing line, per book, per sport, per threshold, per date, cumulative P&L. Personal slip history grades itself. A "Strategy Lab" backtests a strategy against history and stress-tests variance.
10. **Bet sizing.** DFS: flat 0.5 to 1% of bankroll per slip, volume over size. Sportsbooks: per-book Kelly fraction (Full down to 1/16, 1/4 shown as example) in units or dollars, driving a per-line Bet column.

Stack: Next.js frontend, Supabase for auth and slip storage (host `pzniguvsxovzlpfgbdyu.supabase.co` appears in the bundle), Whop for billing and identity, Discord for alerts, a separate React app for the EV calculator. Settings are stored in browser localStorage keys such as `lcslarry_model_global_bettable_v1`, `lcslarry_model_per_bookie_overrides_v1`, `lcslarry_slip_builder_settings_v1`.

### 3.2 What the model is built on (verified statements, per sport)

- **League of Legends.** The "most dialed" model, "the foundation everything else gets built on top of." Rebuilt April 15, 2026 (win rate 60.2% to 62.7%, parlay ROI +30.6% to +53.7% within the month). Props: kills, deaths, assists, per-map (Map 1, Maps 1-2, Maps 1-3) and combos (ShowMaker + Smash Maps 1-2 Kills). Uses team moneyline odds as a feature. Correlation rule: same team, same direction; opposing team, opposite direction.
- **CS2.** Added August 2025 (volume went 700 to 4,100 bets/month). Rebuilt Feb 13, 2026 after edge dulled ("Books adjusted"). Post-rebuild: 46% ROI, 62% win rate first half-month; 29 to 34% ROI in following months. Biggest volume source. Correlation: total rounds drives everything, so all overs or all unders in the same game; PrizePicks nerfs same-team CS2 stacks enough that non-correlated slips are higher EV.
- **Dota 2.** Data-quality pipeline rebuilt Feb 2026 ("up to 50% more bettable lines at 60%+ probability, backtests showing 7% higher accuracy"). Full rebuild July 2026 after a "directional bias — kill projections kept coming in too low." Four days of live data were enough to retune OVER/UNDER scaling.
- **Valorant.** Months of shadow testing, launched Jan 8, 2026. 62% ROI first 3 weeks, 39% Feb, 66% WR Mar, 59.6% Apr, 60.9% May. PrizePicks under-prices same-match Valorant correlation, so stacks are +EV there.
- **Call of Duty.** Three per-map component models (Map 1 Hardpoint, Map 2 SnD, Map 3 Control); Maps 1-3 line modeled as a correlated sum, not an independent sum; combos given their own conservative scaling. Map picks/bans drop about an hour before match and "the model doesn't use the confirmed map yet." Rule: map-agnostic Map 1, Map 3, Maps 1-3 lines become non-bettable if the book moves against the pick. Most COD edge is on UNDERs (May 2026: UNDER 343-243, OVER slightly red).
- **Cross-sport player ban list (Apr 15, 2026).** "Players are only banned if there's statistically significant evidence they're underperforming expectation. The list is fully data-driven and updates over time. Backtested to improve opening accuracy, closing accuracy, and overall P&L." About +0.3% accuracy lift.
- **General observation from the data:** across CS, Dota and COD the edge is skewed toward UNDERs, consistent with books shading kill lines upward toward the public's over bias.

### 3.3 The exact EV math (from ev.lcslarry.com source)

Net-profit-per-unit ladders, indexed by number of legs hit (index 0 = 0 hits). These are the arrays shipped in the calculator:

```
PRIZEPICKS POWER: 2:[-1,-1,2]  3:[-1,-1,-1,5]  4:[-1,-1,-1,-1,9]  5:[-1,-1,-1,-1,-1,19]  6:[-1,-1,-1,-1,-1,-1,36.5]
PRIZEPICKS FLEX:  3:[-1,-1,.25,1.25]  4:[-1,-1,-1,.5,5]  5:[-1,-1,-1,-.6,1,9]  6:[-1,-1,-1,-1,-.6,1,24]
UNDERDOG POWER:   2:[-1,-1,2.5]  3:[-1,-1,-1,5.5]  4:[-1,-1,-1,-1,9]  5:[..,19]  6:[..,34]  7:[..,64]  8:[..,119]
UNDERDOG FLEX:    3:[-1,-1,0,2]  4:[-1,-1,-1,.5,5]  5:[-1,-1,-1,-1,1.5,9]  6:[-1,-1,-1,-1,-.75,1.6,24]  7:[..,-.5,1.75,39]  8:[..,0,2,79]
```

Slip EV with independent legs at a common hit rate p and n legs:

```
EV = sum_{k=0..n} C(n,k) * p^k * (1-p)^(n-k) * net[k]
```

The calculator's Monte Carlo simulator draws Bernoulli(p) per leg, looks up net[k], runs 10,000 paths of (days x bets per day), and reports median / 25th / 75th percentile paths, max drawdown, max run-up, and a "luck factor" (realized minus expected units). Break-even leg hit rates and EV at various hit rates computed from those ladders:

| Book / type / legs | Break-even | EV at 58% | EV at 60% | EV at 62% |
|---|---|---|---|---|
| PP Power 3 | 55.0% | +17.1% | +29.6% | +43.0% |
| PP Power 4 | 56.2% | +13.2% | +29.6% | +47.8% |
| PP Flex 5 | 54.3% | +26.9% | +43.4% | +61.5% |
| PP Flex 6 | 54.2% | +40.2% | +66.4% | +96.6% |
| UD Power 3 | 53.6% | +26.8% | +40.4% | +54.9% |
| UD Flex 6 | 53.8% | +45.7% | +72.9% | +104.3% |

For the slip builder, legs have different probabilities, so the real implementation uses a Poisson-binomial (convolution over per-leg probabilities) rather than the common-p shortcut, and for correlated stacks it substitutes the joint probability of the stack for the product of marginals (the Correlated Stacks page exposes those joint numbers, e.g. two 60% CS2 legs on the same team: 40.6% joint vs 36% independent).

### 3.4 Betting rules the product enforces or teaches

- Only bet lines at or above 60% model probability; below that "you're grinding against the vig."
- Bet the opening line. Turn on alerts. If a line moved more than 10% from open, skip it "even if the model says there's value left."
- Volume is the product: hundreds of slips a month, 0.5 to 1% of bankroll per slip, never more than 20% of bankroll on a slate.
- Diversify across matches, players, and titles; exposure caps per player and per game are built in.
- Prefer 3-leg for low variance, 5/6 flex for max EV; simulate first.
- Correlation: lean in on Valorant; avoid in CS2; check the nerf for LoL/Dota/COD; never mix overs and unders from the same CS2/Valorant map.
- Track everything (Pikkit integration recommended); judge by months, not days.
- Seasonality: Jan to Jun has the most matches, the sloppiest lines, and the best ROI (Feb 2025 +119%, Apr 2025 +88%); December is the dead zone.

---

## 4. Model evolution timeline (what actually moved the needle)

| Date | Change | Effect reported |
|---|---|---|
| 2023 | Larry starts building LoL models, bets personally | "six figures," limited everywhere |
| Aug 2024 | Public Underdog picks with Edge/Multiplier table "sourced from our in-house AI models" | |
| Apr 2025 | Product launch via Whop with Juiced Bets | 43% ROI on first 2,000 bets |
| Jul 2025 | Opening vs closing tracking published | 56% ROI at open, 22% at close |
| Aug 2025 | CS2 model ships | Volume 700 to 4,100/month |
| Dec 2025 | First red month (low volume, sharper lines) | -11% |
| Jan 8 2026 | Valorant model live after months of shadow testing | +62% ROI first 3 weeks |
| Feb 2026 | CS2 rebuild, Dota data pipeline, COD from scratch, Thunderpick straight betting | Best month ever, +31.5% |
| Apr 6 2026 | Bovada and Stake live | Stake +4.4% ROI on huge volume |
| Apr 15 2026 | New LoL model; statistical player ban list | LoL parlay ROI 30% to 54%; +0.3% accuracy |
| May 26 2026 | Rebet, Chalkboard; slip tracker moved server-side | 13 books |
| Jul 2026 | Dota rebuild + OVER/UNDER recalibration; COD rebuild with correlated Maps 1-3 math; bumped-line rule | Dota open 65.4%, close 68.6% (small sample) |
| In progress | Polymarket esports markets scraped; model pending; "something aimed at the closing-line problem" | |

Pattern: every new market prints hardest in its first 1 to 3 months and then decays as books adjust (Valorant 62% to 39% to 25% ROI). Larry's counter is continuous rebuilds, more books, and more sports.

---

## 5. Replication blueprint

Legend: [V] verified from Larry's materials, [I] inferred / standard practice, [U] unknown.

### 5.1 Data you need

**Book lines (the target side).** [V] Larry scrapes 13 books once a minute and stores open + current. Public, unofficial endpoints exist for the two that matter most:
- PrizePicks: `https://api.prizepicks.com/projections?league_id=<id>&per_page=...` (JSON:API; esports league IDs are discoverable from `/leagues`; the endpoint returned HTTP 403 to a bare curl from this container, so expect to need browser-like headers or a residential IP). Apify and similar hosted scrapers cover it for a fee.
- Underdog: `https://api.underdogfantasy.com/beta/v5/over_under_lines` (public, read-only; several GitHub scrapers, e.g. aidanhall21/underdog-fantasy-pickem-scraper).
- Sportsbooks (Thunderpick, Stake, Bovada) and Polymarket for team moneylines and straight props; OddsPapi or The Odds API can supply consensus/Pinnacle-style match odds, which Larry uses as a model feature. [V that ML odds are a feature; I on source]

**Match/player history (the feature side).**
- LoL: Oracle's Elixir CSVs (per-player per-game rows for all major leagues since 2014, includes kills/deaths/assists, side, patch, game length, team gold). Leaguepedia / lol.fandom Cargo API for rosters and schedules. [I: the obvious choice; Larry's source is U]
- CS2: HLTV (Cloudflare-protected; scrapers exist using Playwright/undetected-chromedriver), or parse demos. Need per-map per-player kills, rounds played, map, opponent, event tier.
- Dota 2: OpenDota API (free tier ~60 req/min, 50k/month; `/proMatches`, `/matches/{id}` gives per-player kills, duration).
- Valorant: vlr.gg via unofficial REST APIs (axsddlr/vlrggapi and forks) for per-map K/D/A, agents, map veto.
- COD: breakingpoint.gg (has an API link) or Cito CDL Stats API for per-map, per-mode kills.

**Storage.** Postgres (Supabase, as Larry uses) with tables: players, teams, matches, maps (player-map stat rows), lines (book, prop, player, scope, open, current, timestamp), predictions (projection, prob, EV, version), grades.

### 5.2 Modeling approach

[V] The model is two parts: (a) a projection of the stat for the exact scope of the prop (Map 1, Maps 1-2, Maps 1-3, series, combo), (b) a probability of the OVER/UNDER given the projection and the line, calibrated per sport, per direction, per map, per line type.

[I] A defensible reconstruction that matches everything Larry has said:

1. **Per-map rate model.** For each player and stat, model kills per map as a function of: player's recent per-map rate (exponentially weighted, last 20 to 40 maps), role/position, team's kill-share structure, opponent's kills-allowed and pace (LoL: game length and team kill totals; CS2/Valorant: expected rounds, which is a function of team strength gap; Dota: duration; COD: mode-specific rates), team win probability from moneyline odds (kills are strongly conditional on winning in MOBAs), tournament tier, patch/meta era, side, and roster changes. Gradient-boosted trees (LightGBM/XGBoost) on a few hundred thousand player-map rows are the practical choice and match the "hundreds of thousands of data points" language. A hierarchical/Bayesian rate model is a strong alternative for small samples (new rosters, secondary regions).
2. **Distribution, not just a mean.** Kills are over-dispersed counts; use negative binomial (or a quantile/distributional GBM) so you can compute P(kills > line) directly. Larry's "probability scaling" step suggests he outputs a projection and then maps (projection - line, plus context) to a probability with a learned monotone curve (isotonic or Platt-style) fit on graded history. Either route works; the calibration step is what matters.
3. **Series scopes.** Maps 1-2 / Maps 1-3 are sums of correlated per-map counts. [V] Larry moved COD from an independent sum to a correlated one. Model the per-map draws with a shared player-form factor (or estimate the within-series correlation empirically and inflate variance), and account for match format (BO3 may end 2-0, so "Maps 1-3" props often void or settle on 2 maps; many books void map-3 props, which is why the "skip voidable late-map props" filter exists).
4. **Combos (player A + player B kills).** Joint distribution with the empirical same-team correlation; never multiply marginals. [V]
5. **Calibration loop.** Bucket predictions by predicted probability (55-60, 60-65, ...) and by OVER/UNDER, sport, map, line type; compare to realized hit rate; retune the scaling monthly and on any structural change (patch, new season). [V] Larry retuned Dota on 4 days of data when the miss was one-directional.
6. **Ban list.** Flag players whose realized hit rate vs model is significantly below expectation (e.g. binomial test at alpha 0.05 on the last N graded lines); exclude their lines. [V]
7. **Line-movement features.** Store opening line and time-since-open. Treat a move against you of more than about 10% as a signal the market disagrees; either exclude (Larry's rule) or feed movement into the probability model as a market-information feature (closing-line accuracy is the harder target; Larry's closing ROI is roughly a third of his opening ROI).

### 5.3 Decision layer

1. Per-line EV for DFS: `EV_leg = p * (odds_leg - 1) - (1 - p)` with `odds_leg = payout^(1/4)` for the book's 4-pick power (1.78 PP, 1.86 UD). [V]
2. Bettable if `p >= 0.60` and `EV_leg >= 0.05` and exposure caps not hit. [V defaults]
3. Slip builder: choose the highest-EV combination of bettable, unused lines under constraints (max uses per player, per game, per book; slip type/size), computing slip EV by Poisson-binomial over the payout ladder; for candidate correlated pairs, replace marginal product with joint probability and compare EV against the book's reduced payout. Greedy plus local swaps is enough; it need not be globally optimal. [V behaviour, I algorithm]
4. Deep links: PrizePicks `?projections=<id>-<o|u>-<line>` per leg. [V]
5. Sizing: flat 0.5 to 1% of bankroll per DFS slip; fractional Kelly for straight books. [V]
6. Grading: pull final box scores from the same stat sources, settle each line under the book's rules (void on DNP, map not played), store open-line and close-line grades separately.

### 5.4 Minimal viable build order

1. Week 1-2: PrizePicks + Underdog scrapers with opening/current line capture; LoL data from Oracle's Elixir; Postgres schema.
2. Week 3-4: LoL kills/deaths/assists per-map model; probability calibration; grade against last 3 months of captured lines (you cannot backtest without your own line history, so start capturing lines on day one).
3. Week 5: EV + threshold + simple slip builder + Discord webhook alerts.
4. Week 6+: Add CS2 (largest volume), then Valorant (softest lines per Larry), then Dota, then COD. Add correlation modeling once you have graded same-game pairs to estimate joint hit rates.

### 5.5 Risks and unknowns

- [U] Larry's exact features, learners, and scaling method are undisclosed; expect a long tail of tuning before hitting a calibrated 60% at volume. His own CS2 model needed a full rebuild within six months as books adjusted.
- [V] Books limit winners (PrizePicks ~$25/day, Underdog ~$75/day cited). Thunderpick cancelled Larry's affiliate deal because his referred users were profitable.
- [V] Edge decays fast after open; alert latency and scrape cadence (one minute) are part of the edge.
- Scraping book APIs may violate their terms; PrizePicks returned 403 to a plain request during this research.
- [V] Realized member results vary widely with sizing discipline and timing; the tracker is theoretical at open.

---

## 6. Sources

Site: https://www.lcslarry.com/ , /about , /faq , /onboarding , /results , /terms
Blog: /blog/welcome-to-lcslarry , /blog/understanding-ev-betting , /blog/esports-model-worth-it , /blog/opening-closing-lines-betting-model , /blog/top-5-mistakes-bettors-make , /blog/prime-time-esports-betting , /blog/2025-results , /blog/january-2026-recap , /blog/february-2026-recap , /blog/march-2026-recap , /blog/april-2026-recap , /blog/may-2026-recap , /blog/how-to-make-money-on-prizepicks-esports-2026 , /blog/correlating-esports-props-dfs , /blog/dota-cod-model-update-july-2026
App bundles: https://www.lcslarry.com/_next/static/chunks/* (tutorial, help-center, settings, results methodology strings); https://ev.lcslarry.com/assets/index-CqyaS3-G.js (payout ladders, EV and simulator functions)
X: https://x.com/LCSLarry ; tweets 1751079004692480044 (Jan 2024), 1821759056413143551 (Aug 2024 Underdog table), 1907886835160002826 (launch), 1998426020500951470 (slip screenshot with Prob/EV columns), 2073794819739341082 (Peyz POTD graphic: hit probability 61.8%, EV +12.2%, model line 16.5 vs book 19.5), 2088729134579249423 (Aug 2026 results graphic)
Whop: https://whop.com/discover/juiced-bets-vip/esports-model/ (4.9 stars, 1,075 reviews, $99.99 per 2 weeks)
Data-source references: https://oracleselixir.com/tools/downloads , https://github.com/axsddlr/vlrggapi , https://github.com/aidanhall21/underdog-fantasy-pickem-scraper , https://github.com/StrandedPond/hltv_scraper , https://breakingpoint.gg/stats , https://citoapi.com/cdl-stats-api/ , https://oddspapi.io/ , OpenDota API docs
