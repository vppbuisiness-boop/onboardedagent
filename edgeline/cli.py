"""edgeline command line interface."""
from __future__ import annotations

import datetime as dt
import json
import time

import pandas as pd
import typer

from . import db
from .config import DB_PATH, DEFAULT_MARKET_SHRINK, DEFAULT_MAX_LINE_MOVE, DEFAULT_MIN_EV, DEFAULT_MIN_PROB

app = typer.Typer(help="Esports player-prop pricing engine (line capture, model, calibration, EV, slips, grading).", no_args_is_help=True)
lines_app = typer.Typer(help="Book line capture and inspection.", no_args_is_help=True)
history_app = typer.Typer(help="Historical match data loaders.", no_args_is_help=True)
model_app = typer.Typer(help="Train and inspect models.", no_args_is_help=True)
slips_app = typer.Typer(help="Build and manage slips.", no_args_is_help=True)
ev_app = typer.Typer(help="EV tables and bankroll simulation.", no_args_is_help=True)
banlist_app = typer.Typer(help="Data-driven player ban list.", no_args_is_help=True)
alerts_app = typer.Typer(help="Discord alerts for new bettable lines.", no_args_is_help=True)
app.add_typer(lines_app, name="lines")
app.add_typer(history_app, name="history")
app.add_typer(model_app, name="model")
app.add_typer(slips_app, name="slips")
app.add_typer(ev_app, name="ev")
app.add_typer(banlist_app, name="banlist")
app.add_typer(alerts_app, name="alerts")

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)


@app.command()
def init():
    """Create the SQLite database and tables."""
    with db.session() as conn:
        n = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'").fetchone()[0]
    typer.echo(f"database ready at {DB_PATH} ({n} tables)")


@lines_app.command("pull")
def lines_pull(sports: str = typer.Option("lol,cs2,val,dota,cod", help="comma-separated"), book: str = "prizepicks"):
    """Snapshot the current board; first sighting of a projection becomes its opening line."""
    from .books import prizepicks, sleeper

    if book not in ("prizepicks", "sleeper"):
        raise typer.BadParameter("pulls are implemented for prizepicks and sleeper (underdog needs client headers)")
    with db.session() as conn:
        summary = prizepicks.pull(conn, [s.strip() for s in sports.split(",") if s.strip()]) if book == "prizepicks" else sleeper.pull(conn)
    for sport, s in summary.items():
        line = f"{sport:5s} projections={s['projections']:4d} new={s['new']:4d} updated={s['updated']:4d} moved={s['moved']:3d}"
        typer.echo(line + (f"  ERROR {s['error']}" if s.get("error") else ""))


@lines_app.command("shop")
def lines_shop(sport: str | None = None, out: str | None = typer.Option(None, help="write the comparison to this CSV"), limit: int = 40):
    """Same player and stat across books: each book's line, the model's lean and EV on each, and the best book."""
    from .books.shop import shop

    with db.session() as conn:
        df = shop(conn, sport)
    if df.empty:
        typer.echo("no player posted by more than one book")
        return
    pd.set_option("display.width", 220)
    typer.echo(df.head(limit).to_string(index=False))
    if out:
        df.to_csv(out, index=False)


@lines_app.command("watch")
def lines_watch(sports: str = "lol,cs2,val,dota,cod", interval: int = 60, iterations: int = 0,
                price: bool = typer.Option(True, help="price every trained sport's board right after each pull so no line starts unpriced"),
                alert: bool = typer.Option(False, help="send Discord alerts for new bettable lines (EDGELINE_DISCORD_WEBHOOK)"),
                reprice_every: int = typer.Option(1800, help="seconds between full re-pricing passes when the board has not changed"),
                books: str = typer.Option("prizepicks,sleeper", help="comma-separated books to pull and price each cycle")):
    """Poll the board every `interval` seconds (0 iterations = forever), pricing new lines as they post.

    Pricing rebuilds each sport's full as-of state, which is the expensive part, so a sport is re-priced only
    when its board gained or moved lines, plus one full pass every `reprice_every` seconds so retrained models
    take effect."""
    from .books import prizepicks, sleeper

    book_list = [b.strip() for b in books.split(",") if b.strip()]
    i = 0
    last_full = 0.0
    while True:
        try:
            with db.session() as conn:
                summary = prizepicks.pull(conn, [s.strip() for s in sports.split(",")]) if "prizepicks" in book_list else {}
                if "sleeper" in book_list:
                    try:
                        for sp, s in sleeper.pull(conn).items():
                            summary[sp] = {k: summary.get(sp, {}).get(k, 0) + s.get(k, 0) for k in ("projections", "new", "updated", "moved")}
                    except Exception as exc:  # a second book must never stop the primary pull
                        typer.echo(f"sleeper pull failed: {exc}")
            moved = sum(s["moved"] for s in summary.values())
            new = sum(s["new"] for s in summary.values())
            typer.echo(f"{dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} new={new} moved={moved}")
            if price or alert:
                from .models.predict import load_models, price_board

                full = (time.time() - last_full) >= reprice_every
                todo = [s.strip() for s in sports.split(",") if full or (summary.get(s.strip(), {}).get("new", 0) + summary.get(s.strip(), {}).get("moved", 0)) > 0]
                if full:
                    last_full = time.time()
                priced, bettable = 0, 0
                with db.session() as conn:
                    for sport in todo:
                        if load_models(sport):
                            for bk in book_list:
                                try:
                                    out = price_board(conn, sport, bk)
                                except Exception as exc:  # a sport without history should not stop the loop
                                    typer.echo(f"pricing {sport}/{bk} failed: {exc}")
                                    continue
                                priced += int(out["projection"].notna().sum()) if not out.empty else 0
                                bettable += int(out["bettable"].sum()) if not out.empty else 0
                    if todo:
                        typer.echo(f"priced {priced} lines, {bettable} bettable ({'full pass' if full else ', '.join(todo)})")
                    if alert:
                        from .alerts import send

                        n = send(conn)
                        if n:
                            typer.echo(f"alerted {n} new bettable lines")
        except Exception as exc:  # keep polling on transient errors
            typer.echo(f"pull failed: {exc}")
        i += 1
        if iterations and i >= iterations:
            break
        time.sleep(interval)


@lines_app.command("underdog-dump")
def lines_underdog_dump(out: str = "data/raw/underdog_over_under_lines.json"):
    """Fetch Underdog's raw over/under lines JSON using headers from EDGELINE_UNDERDOG_HEADERS and save it.

    Capture the headers once in your browser: open underdogfantasy.com, DevTools > Network, click any request to
    api.underdogfantasy.com, and copy the client-type, client-version and client-device-id request headers into
    EDGELINE_UNDERDOG_HEADERS as a JSON object. The saved file is the input for building the Underdog parser.
    """
    from .books.underdog import fetch_raw

    payload = fetch_raw()
    from pathlib import Path

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(payload))
    keys = list(payload.keys()) if isinstance(payload, dict) else type(payload).__name__
    typer.echo(f"saved {out}; top-level keys: {keys}")


@lines_app.command("show")
def lines_show(sport: str = "lol", book: str = "prizepicks", limit: int = 60, upcoming: bool = True):
    """Show open vs current lines with movement."""
    with db.session() as conn:
        df = pd.read_sql_query("SELECT * FROM lines WHERE book=? AND sport=? ORDER BY start_time, player_name", conn, params=(book, sport))
    if df.empty:
        typer.echo("no lines")
        return
    if upcoming:
        st = pd.to_datetime(df["start_time"], utc=True, errors="coerce")
        df = df[st > pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=1)]
    df["move%"] = ((df["current_line"] - df["open_line"]) / df["open_line"] * 100).round(1)
    cols = ["projection_id", "player_name", "team", "opponent", "stat_type", "open_line", "current_line", "move%", "current_odds_type", "status", "start_time"]
    typer.echo(df[cols].head(limit).to_string(index=False))
    typer.echo(f"{len(df)} lines")


@history_app.command("opendota")
def history_opendota(since: str = typer.Option((dt.date.today() - dt.timedelta(days=365)).isoformat())):
    """Load professional Dota 2 player-match rows from OpenDota's SQL explorer."""
    from .data import opendota

    with db.session() as conn:
        out = opendota.load(conn, dt.date.fromisoformat(since))
    typer.echo(json.dumps(out))


@history_app.command("leaguepedia")
def history_leaguepedia(since: str = typer.Option((dt.date.today() - dt.timedelta(days=365)).isoformat()), until: str | None = None):
    """Load LoL player-game rows from Leaguepedia (paced; rate limited for anonymous use)."""
    from .data import leaguepedia

    with db.session() as conn:
        out = leaguepedia.load(conn, dt.date.fromisoformat(since), dt.date.fromisoformat(until) if until else None)
    typer.echo(json.dumps(out))


@history_app.command("oracles-elixir")
def history_oe():
    """Load any Oracle's Elixir CSVs found in data/raw/."""
    from .data import oracles_elixir

    with db.session() as conn:
        out = oracles_elixir.load_files(conn)
    typer.echo(json.dumps(out))


@history_app.command("vlr")
def history_vlr(pages: int = 10, start_page: int = 1, since: str | None = typer.Option(None, help="stop at matches older than YYYY-MM-DD")):
    """Scrape Valorant per-map player stats from vlr.gg (newest first, ~1.5 s per request)."""
    from .data import vlr

    with db.session() as conn:
        out = vlr.load(conn, pages, start_page, dt.date.fromisoformat(since) if since else None, progress=typer.echo)
    typer.echo(json.dumps(out))


@history_app.command("bo3")
def history_bo3(since: str = typer.Option((dt.date.today() - dt.timedelta(days=120)).isoformat()), until: str | None = None,
                tiers: str | None = typer.Option(None, help="comma-separated, e.g. s,a,b"), max_matches: int | None = None,
                refresh_days: float = typer.Option(2.0, help="re-fetch maps that began within this many days (team links lag)")):
    """Load CS2 per-map player stats from bo3.gg (one request per map)."""
    from .data import bo3

    with db.session() as conn:
        out = bo3.load(conn, dt.date.fromisoformat(since), dt.date.fromisoformat(until) if until else None,
                       tiers.split(",") if tiers else None, max_matches, refresh_days=refresh_days, progress=typer.echo)
    typer.echo(json.dumps(out))


@history_app.command("lolesports")
def history_lolesports(since: str = typer.Option((dt.date.today() - dt.timedelta(days=365)).isoformat()), until: str | None = None,
                       leagues: str | None = typer.Option(None, help="comma-separated league slugs (default: every league Riot lists except TFT)"),
                       workers: int = 6):
    """Load LoL per-game player stats from Riot's official esports API and livestats feed."""
    from .data import lolesports

    with db.session() as conn:
        out = lolesports.load(conn, dt.date.fromisoformat(since), dt.date.fromisoformat(until) if until else None,
                              leagues.split(",") if leagues else None, workers=workers, progress=typer.echo)
    typer.echo(json.dumps(out))


@history_app.command("breakingpoint")
def history_breakingpoint(sport: str = typer.Option("cod", help="cod or cs2"), since: str = typer.Option((dt.date.today() - dt.timedelta(days=365)).isoformat()),
                          until: str | None = None):
    """Load Call of Duty (or CS2) per-map player stats from Breaking Point's public database."""
    from .data import breakingpoint

    with db.session() as conn:
        out = breakingpoint.load(conn, sport, dt.date.fromisoformat(since), dt.date.fromisoformat(until) if until else None)
    typer.echo(json.dumps(out))


@history_app.command("stats")
def history_stats():
    with db.session() as conn:
        df = pd.read_sql_query("SELECT sport, source, COUNT(*) rows, COUNT(DISTINCT game_id) games, COUNT(DISTINCT player_name) players, MIN(date) first, MAX(date) last FROM player_games GROUP BY sport, source", conn)
    typer.echo(df.to_string(index=False) if not df.empty else "no history loaded")


@model_app.command("train")
def model_train(sport: str = "dota", stats: str | None = typer.Option(None, help="comma-separated; default: every stat the sport has data for"),
                valid_frac: float = 0.2, recency_halflife: float | None = typer.Option(None, help="days; weight training rows by 0.5^(age/halflife)")):
    """Train per-map count models (LightGBM Poisson + NB dispersion + copula correlations + isotonic calibration)."""
    from .features.build import build_training_frame
    from .models import props

    with db.session() as conn:
        pg = pd.read_sql_query("SELECT * FROM player_games WHERE sport=?", conn, params=(sport,))
    if pg.empty:
        typer.echo(f"no history for {sport}; run `edgeline history ...` first")
        raise typer.Exit(code=1)
    frame = build_training_frame(pg)
    typer.echo(f"training frame: {len(frame)} rows, {frame['player_name'].nunique()} players, {frame['date'].min().date()} to {frame['date'].max().date()}")
    if stats is None:
        stats = ",".join(st for st in ("kills", "deaths", "assists", "headshots") if pg[st].notna().sum() > 1000)
        typer.echo(f"stats with data: {stats}")
    for stat in [s.strip() for s in stats.split(",")]:
        m = props.train(frame, sport, stat, valid_frac=valid_frac, recency_halflife=recency_halflife)
        path = m.save()
        props.save_metrics(m)
        typer.echo(props.metrics_summary(m))
        typer.echo(f"  saved {path}")


@model_app.command("tune")
def model_tune(sport: str = "dota", stats: str = "kills,deaths", valid_frac: float = 0.2):
    """Small hyperparameter sweep per stat (lowest validation Poisson deviance wins); saves models/artifacts/<sport>_params.json."""
    from .features.build import build_training_frame
    from .models import props

    with db.session() as conn:
        pg = pd.read_sql_query("SELECT * FROM player_games WHERE sport=?", conn, params=(sport,))
    frame = build_training_frame(pg)
    for stat in [s.strip() for s in stats.split(",")]:
        best, results = props.tune(frame, sport, stat, valid_frac=valid_frac)
        typer.echo(f"{sport}/{stat}: best {best}")
        typer.echo(pd.DataFrame(results).to_string(index=False))


@model_app.command("backtest")
def model_backtest(sport: str = "dota", stat: str = "kills", months: int = 5, shrink: float = DEFAULT_MARKET_SHRINK, threshold: float = DEFAULT_MIN_PROB,
                   out: str | None = typer.Option(None, help="write pooled picks to this CSV"),
                   maps: int = typer.Option(1, help="1 = single-map lines; 2 = map 1 + map 2 sums priced with the live copula")):
    """Walk-forward backtest vs synthetic line-setters (naive trailing mean, book-like model). Not realized ROI."""
    from .features.build import build_training_frame
    from .models.backtest import pooled_summary, walk_forward, walk_forward_two_map

    with db.session() as conn:
        pg = pd.read_sql_query("SELECT * FROM player_games WHERE sport=?", conn, params=(sport,))
    frame = build_training_frame(pg)
    runner = walk_forward_two_map if maps == 2 else walk_forward
    summary, pooled = runner(frame, sport, stat, months, shrink, threshold, progress=typer.echo)
    if summary.empty:
        typer.echo("not enough history for a walk-forward backtest")
        return
    fmt = summary.copy()
    for c in ("hit_rate", "ci_low", "ci_high", "parlay4_roi"):
        fmt[c] = (fmt[c] * 100).round(1)
    typer.echo(fmt.to_string(index=False))
    ps = pooled_summary(pooled)
    for c in ("pick_share", "hit_rate", "ci_low", "ci_high", "parlay4_roi", "parlay4_roi_ci_low", "parlay4_roi_ci_high"):
        ps[c] = (ps[c] * 100).round(1)
    typer.echo("pooled (all folds):")
    typer.echo(ps.to_string(index=False))
    typer.echo("Read as edge over a book that prices like the synthetic setter, not as realized ROI on PrizePicks.")
    if out:
        pooled.to_csv(out, index=False)
        typer.echo(f"picks written to {out}")


@model_app.command("metrics")
def model_metrics(sport: str = "dota", stat: str = "kills"):
    from .models import props

    m = props.PropModel.load(sport, stat)
    typer.echo(props.metrics_summary(m))


@app.command()
def replay(sport: str = "cs2", cutoff: str = typer.Option(..., help="UTC timestamp; models and state use only games before it, lines settled after it are priced at open"),
           book: str = "prizepicks", out: str | None = typer.Option(None, help="write the priced lines to this CSV")):
    """Out-of-sample replay of captured, settled lines with models that know nothing after the cutoff."""
    from .models.replay import replay as _replay, summarize

    with db.session() as conn:
        df = _replay(conn, sport, cutoff, book, progress=typer.echo)
    if df.empty:
        typer.echo("no settled lines after the cutoff")
        return
    summ = summarize(df)
    for c in ("hit_rate", "ci_low", "ci_high"):
        summ[c] = (summ[c] * 100).round(1)
    typer.echo(summ.to_string(index=False))
    if out:
        df.to_csv(out, index=False)


@app.command()
def predict(sport: str = "dota", book: str = "prizepicks", min_prob: float = DEFAULT_MIN_PROB, min_ev: float = DEFAULT_MIN_EV,
            max_move: float = DEFAULT_MAX_LINE_MOVE, include_voidable: bool = False, show_all: bool = False,
            market_shrink: float = typer.Option(DEFAULT_MARKET_SHRINK, help="weight on the book line as a prior for each component mean (0 = pure model)"),
            include_unproven: bool = typer.Option(False, help="also flag markets the backtest does not support (CS2 kills, COD) as bettable")):
    """Price the current board and flag bettable lines."""
    from .models.predict import price_board

    with db.session() as conn:
        out = price_board(conn, sport, book, min_prob, min_ev, max_move, skip_voidable=not include_voidable, market_shrink=market_shrink,
                          include_unproven=include_unproven)
    if out.empty:
        typer.echo("no lines to price")
        return
    priced = out[out["projection"].notna()]
    cols = ["projection_id", "player_name", "team", "opponent", "stat_type", "open_line", "line", "projection", "p_over", "p_under", "lean", "prob", "ev", "bettable", "notes"]
    view = priced if show_all else priced[priced["bettable"] == 1]
    view = view.sort_values(["bettable", "ev"], ascending=False)
    fmt = view[cols].copy()
    for c in ("p_over", "p_under", "prob", "ev"):
        fmt[c] = (fmt[c] * 100).round(1)
    fmt["projection"] = fmt["projection"].round(2)
    typer.echo(fmt.to_string(index=False))
    unm = out[out["projection"].isna()]
    typer.echo(f"\n{len(out)} lines: {len(priced)} priced, {int(priced['bettable'].sum())} bettable, {len(unm)} unpriced (player not in history)")
    if len(unm):
        typer.echo("unmapped players: " + ", ".join(sorted(set(unm["player_name"]))[:40]))


@slips_app.command("build")
def slips_build(book: str = "prizepicks", slip_type: str = "POWER", size: int = 3, max_slips: int = 8, sports: str | None = None,
                max_per_game: int = 1, rank_by: str = "ev", save: bool = True):
    """Assemble EV-optimal slips from bettable predictions."""
    from .slips.builder import build, format_slip, save_slips

    with db.session() as conn:
        slips = build(conn, book, slip_type.upper(), size, max_slips, sports.split(",") if sports else None, max_per_game=max_per_game, rank_by=rank_by)
        if save and slips:
            save_slips(conn, slips)
    if not slips:
        typer.echo("no slips (not enough bettable legs). Run `edgeline predict` first or loosen thresholds.")
        return
    for i, s in enumerate(slips, 1):
        typer.echo(format_slip(s, i))


@slips_app.command("use")
def slips_use(projection_ids: str):
    """Mark projection ids as used so they are excluded from future slips (comma-separated)."""
    from .slips.builder import mark_used

    with db.session() as conn:
        mark_used(conn, "prizepicks", [p.strip() for p in projection_ids.split(",") if p.strip()])
    typer.echo("marked used")


@app.command()
def grade(sport: str = "dota", book: str = "prizepicks", min_age_hours: float = 4.0):
    """Settle finished lines against loaded history rows."""
    from .grading.grade import grade_lines

    with db.session() as conn:
        out = grade_lines(conn, sport, book, min_age_hours)
    typer.echo(json.dumps(out))


@app.command()
def results(book: str = "prizepicks", by: str | None = typer.Option(None, help="group by: sport | lean | stat_type"), min_prob: float = 0.0,
            bettable_only: bool = False):
    """Hit rate, leg ROI and 4-pick parlay ROI against opening and current lines."""
    from .grading.results import results_frame, summarize

    with db.session() as conn:
        df = results_frame(conn, book, min_prob, bettable_only)
    if df.empty:
        typer.echo("no graded predictions yet")
        return
    typer.echo(summarize(df, book, by).to_string(index=False))


@app.command()
def stacks(sport: str = "dota", book: str = "prizepicks", min_prob: float = 0.55, bettable_only: bool = False, limit: int = 30):
    """Price correlated same-game pairs: joint hit probability, lift over independence, break-even 2-pick multiplier."""
    from .models.predict import BoardPricer
    from .models.stacks import find_stacks

    with db.session() as conn:
        pricer = BoardPricer(conn, sport, book)
        pricer.price(store=False)
        df = find_stacks(pricer, min_prob, bettable_only)
    if df.empty:
        typer.echo("no pairs (need two priced legs of the same stat in one game)")
        return
    view = df.head(limit).copy()
    for c in ("joint", "indep", "lift", "ev_std"):
        view[c] = (view[c] * 100).round(1)
    view["breakeven_mult"] = view["breakeven_mult"].round(2)
    typer.echo(view[["game", "leg_a", "leg_b", "relation", "direction", "joint", "indep", "lift", "breakeven_mult", "ev_std", "ids"]].to_string(index=False))
    typer.echo("joint/indep/lift/ev_std in %; ev_std = EV as a 2-pick POWER at the standard multiplier; the stack beats the app if its shaded multiplier > breakeven_mult")


@banlist_app.command("update")
def banlist_update(book: str = "prizepicks", min_n: int = 15, alpha: float = 0.05):
    """Recompute the ban list from graded picks (one-sided binomial test vs the model's own probabilities)."""
    from .models.banlist import update

    with db.session() as conn:
        table = update(conn, book, min_n, alpha)
    if table.empty:
        typer.echo("no graded picks yet")
        return
    typer.echo(table.head(30).to_string(index=False))
    typer.echo(f"banned: {int(table['banned'].sum())} of {len(table)} players with graded picks")


@banlist_app.command("show")
def banlist_show():
    with db.session() as conn:
        df = pd.read_sql_query("SELECT * FROM banned_players ORDER BY pvalue", conn)
    typer.echo(df.to_string(index=False) if not df.empty else "ban list is empty")


@alerts_app.command("send")
def alerts_send(book: str = "prizepicks", dry_run: bool = False):
    """Send Discord alerts for bettable lines not yet alerted (EDGELINE_DISCORD_WEBHOOK)."""
    from .alerts import send

    with db.session() as conn:
        n = send(conn, book, dry_run=dry_run)
    typer.echo(f"{n} lines alerted" if n else "nothing new to alert")


@app.command()
def roi(book: str = "prizepicks", min_prob: float = 0.0):
    """Hit rate on settled opening lines with a 95% interval, and the 4-pick parlay ROI it implies (LCSLarry's convention)."""
    from .grading.results import results_frame
    from .grading.roi import format_gauge, gauge

    with db.session() as conn:
        df = results_frame(conn, book, min_prob)
    if df.empty:
        typer.echo("no graded predictions yet")
        return
    typer.echo(format_gauge(gauge(df, book, "all model leans")))
    typer.echo(format_gauge(gauge(df[df["bettable"] == 1], book, "bettable picks (>=60% prob, >=5% EV)")))
    for sport, g in df.groupby("sport"):
        typer.echo(format_gauge(gauge(g, book, f"{sport} (all leans)")))


@app.command()
def export():
    """Write captured lines, snapshots, predictions, grades and slips to data/exports/*.csv (keep these in git)."""
    from .export import EXPORT_DIR, export_tables

    with db.session() as conn:
        counts = export_tables(conn)
    typer.echo(f"exported to {EXPORT_DIR}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))


@app.command("import")
def import_():
    """Restore exported tables into the database (INSERT OR IGNORE)."""
    from .export import EXPORT_DIR, import_tables

    with db.session() as conn:
        counts = import_tables(conn)
    typer.echo(f"imported from {EXPORT_DIR}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))


@ev_app.command("table")
def ev_table(book: str = "prizepicks", slip_type: str = "POWER", size: int = 4):
    """EV of a slip type across per-leg hit rates, plus break-even."""
    from .ev.math import breakeven_hit_rate, ev_table as _table
    from .ev.payouts import ladder

    net = ladder(book, slip_type.upper(), size)
    typer.echo(f"{book} {slip_type.upper()} {size}-pick ladder(net)={net} break-even leg hit rate={breakeven_hit_rate(net):.1%}")
    for h, e in _table(book, slip_type.upper(), size, [0.52, 0.55, 0.58, 0.60, 0.62, 0.65, 0.70]):
        typer.echo(f"  hit {h:.0%} -> EV {e:+.1%}")


@ev_app.command("simulate")
def ev_simulate(hit_rate: float = 0.60, days: int = 30, bets_per_day: int = 4, size: int = 4, book: str = "prizepicks",
                slip_type: str = "POWER", iterations: int = 2000, unit: float = 10.0):
    """Monte Carlo bankroll simulation (median / p25 / p75 paths)."""
    from .ev.simulate import simulate

    r = simulate(hit_rate, days, bets_per_day, size, book, slip_type.upper(), iterations, seed=7)
    for k in ("p25", "median", "p75"):
        s = r[k]
        typer.echo(f"{k:6s}: final {s.final*unit:+9.2f} ({s.final:+.1f} u) roi/bet {s.roi_per_bet:+.1%} max drawdown {s.max_drawdown*unit:.2f} props {s.prop_hits}/{s.total_props}")
    typer.echo(f"mean final {r['mean_final']*unit:+.2f}  P(profit)={r['prob_profit']:.1%}  p90 drawdown {r['worst_drawdown_p90']*unit:.2f}")


if __name__ == "__main__":
    app()
