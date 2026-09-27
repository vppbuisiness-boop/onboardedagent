# oddsfloor.com integration

`edgeline web --bankroll 2000` writes three JSON feeds to `data/web/v1/` in the shapes oddsfloor's board already
speaks (`opportunities[]` with bookKey, sportKey, eventId, commenceTime, homeTeam, awayTeam, market, participant,
outcome, point, decimalOdds, fairProb, ev, tier, kelly, link). The 3-hourly forward-test pass rewrites and pushes
them, so the branch is the live source:

| Feed | What it holds | URL |
|---|---|---|
| `esports-ev.json` | one opportunity per upcoming standard line with a lean (`tier` = bettable / lean / gated / voidable, `inHorizon` = today or tomorrow in New York) | `https://raw.githubusercontent.com/vppbuisiness-boop/onboardedagent/claude/amazing-ritchie-q2np39/data/web/v1/esports-ev.json` |
| `esports-slips.json` | the best slip at every price (POWER 2-6, FLEX 3-6) with EV, P(top), P(paid), quarter-Kelly stake, legs and a PrizePicks deep link | `.../data/web/v1/esports-slips.json` |
| `esports-record.json` | the record on real captured lines per slice: wins, losses, hit rate, Wilson interval, matches, match-cluster interval, break-even and target legs | `.../data/web/v1/esports-record.json` |

GitHub serves the raw files with `access-control-allow-origin: *`, so a static page can fetch them cross-origin
with no token. Every field is documented in `edgeline/web.py`.

## Drop-in page

`web/esports.html` is a complete page in the site's shell (same header markup, `/sharpline.css` tokens, `/theme.js`
and `/nav.js`). It renders the best slips as cards, the board with sport / tier / minimum-probability / horizon
filters, and the record table with the break-even and target lines. It refreshes itself every five minutes and
flags the feeds as stale when the last rebuild is more than four hours old.

1. Copy `web/esports.html` into the Vercel project so it serves at `/esports` (the same way `/ev` and `/screen` are served).
2. Add the route to the rail and the top nav in `nav.js` (`Esports Props` under *Find bets*).
3. Optional, recommended: serve the feeds same-origin so they cache on Vercel's edge and the page never depends on
   GitHub being reachable from the browser. Add to `vercel.json`:

   ```json
   {
     "rewrites": [
       {
         "source": "/v1/esports-:name.json",
         "destination": "https://raw.githubusercontent.com/vppbuisiness-boop/onboardedagent/claude/amazing-ritchie-q2np39/data/web/v1/esports-:name.json"
       }
     ],
     "headers": [
       { "source": "/v1/esports-:name.json", "headers": [ { "key": "Cache-Control", "value": "s-maxage=300, stale-while-revalidate=3600" } ] }
     ]
   }
   ```

   then set `window.ESPORTS_FEED_BASE = "/v1"` in the page's first script block. Until then the page reads the
   GitHub URL directly; `?feed=<base>` on the URL overrides both for testing.

## Not done on purpose

The feeds are static JSON on the branch, not rows in the site's Supabase project. Writing into that project
(new tables `esports_opportunities`, `esports_slips`, `esports_record` with public-select policies, or an edge
function that mirrors the branch) is a one-hour change but it touches the production database, so it waits for
an explicit go-ahead.
