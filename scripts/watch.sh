#!/usr/bin/env bash
# Start (or restart) the edgeline line watcher in the background.
# Usage: scripts/watch.sh [interval_seconds] [sports]
# Env:   EDGELINE_DISCORD_WEBHOOK to enable alerts.
set -euo pipefail
cd "$(dirname "$0")/.."
INTERVAL="${1:-120}"
SPORTS="${2:-lol,cs2,val,dota,cod}"
mkdir -p data/logs
# kill any previous watcher without matching this script's own command line
for pid in $(pgrep -f "edgeline lines watch" || true); do
  [ "$pid" != "$$" ] && kill "$pid" 2>/dev/null || true
done
sleep 1
ALERT_FLAG=""
[ -n "${EDGELINE_DISCORD_WEBHOOK:-}" ] && ALERT_FLAG="--alert"
nohup edgeline lines watch --sports "$SPORTS" --interval "$INTERVAL" $ALERT_FLAG >> data/logs/watch.log 2>&1 &
sleep 2
echo "watcher pid(s): $(pgrep -f 'edgeline lines watch' | tr '\n' ' ')"
echo "log: data/logs/watch.log"
