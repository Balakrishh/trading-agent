#!/usr/bin/env bash
# migrate_to_pi.sh — copy the agent's runtime state, secrets and Claude Code
# context from this Mac to the Raspberry Pi. Run on the Mac, from the repo.
# Runbook: docs/runbooks/08_raspberry_pi_deployment.md §6.
#
#   scripts/migrate_to_pi.sh                 # dry run: lists what would copy
#   scripts/migrate_to_pi.sh --go            # copies
#
# Env overrides: PI_HOST (balakrishh@myrasberrypi.local),
#                PI_REPO (/home/balakrishh/Documents/trading-agent)
#
# It refuses to run while the Mac agent is running: two agents on one
# Alpaca account would double-trade, and two copies of the Schwab token
# invalidate each other (the refresh token rotates on every use).
set -euo pipefail

PI_HOST="${PI_HOST:-balakrishh@myrasberrypi.local}"
PI_REPO="${PI_REPO:-/home/balakrishh/Documents/trading-agent}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GO=0; [ "${1:-}" = "--go" ] && GO=1

if pgrep -f "trading_agent.agent_supervisor|trading_agent.agent( |$)|trading_agent.data_server|streamlit run" >/dev/null; then
    echo "The agent, its supervisor or Streamlit is still running on this Mac."
    echo "Stop them first (runbook §6.1):"
    echo "  launchctl bootout gui/\$(id -u)/com.trading-agent.headless"
    echo "  launchctl bootout gui/\$(id -u)/com.trading-agent.daily-reviewer"
    echo "  pkill -f 'streamlit run'; pkill -f trading_agent.data_server"
    exit 1
fi

if pgrep -f "trading_agent.mcp" >/dev/null; then
    echo "Note: a Claude Code session on this Mac has the trading-agent MCP server open."
    echo "      After the copy, quit it: its quote tools use the Schwab token, and a"
    echo "      refresh from the Mac would invalidate the Pi's copy."
fi

RSYNC=(rsync -az --human-readable --itemize-changes)
[ $GO -eq 1 ] || RSYNC+=(--dry-run)

echo "== Runtime state → ${PI_HOST}:${PI_REPO}"
cd "$REPO"
STATE=()
for p in trade_journal trade_plans daily_reviews knowledge_base pending_orders \
         pending_preset_updates journal_kb logs STRATEGY_PRESET.json AGENT_LOG .env \
         .claude/settings.local.json docs/plans/journal_learnings.md; do
    [ -e "$p" ] && STATE+=("$p")
done
"${RSYNC[@]}" --relative "${STATE[@]}" "${PI_HOST}:${PI_REPO}/" | tail -5

echo "== Schwab OAuth token → ${PI_HOST}:~/.schwab_tokens.json"
if [ -f "$HOME/.schwab_tokens.json" ]; then
    "${RSYNC[@]}" "$HOME/.schwab_tokens.json" "${PI_HOST}:.schwab_tokens.json"
else
    echo "  (none on this Mac — run 'python -m trading_agent.schwab_oauth login' on the Pi)"
fi

# Claude Code keys project history by the folder path with '/' → '-'.
SRC_KEY="$(printf '%s' "$REPO" | tr '/' '-')"
DST_KEY="$(printf '%s' "$PI_REPO" | tr '/' '-')"
echo "== Claude Code context: projects/${SRC_KEY} → projects/${DST_KEY}"
ssh "$PI_HOST" "mkdir -p ~/.claude/projects/${DST_KEY}"
"${RSYNC[@]}" "$HOME/.claude/projects/${SRC_KEY}/" "${PI_HOST}:.claude/projects/${DST_KEY}/" | tail -5
# Each saved message records the Mac's working directory; /resume on the Pi
# only lists conversations whose directory matches, so point them at PI_REPO.
if [ $GO -eq 1 ]; then
    ssh "$PI_HOST" "sed -i 's#\"cwd\":\"${REPO}\"#\"cwd\":\"${PI_REPO}\"#g' ~/.claude/projects/${DST_KEY}/*.jsonl"
fi

[ $GO -eq 1 ] && echo "Done. Next: runbook §7 (start services on the Pi)." \
              || echo "Dry run only. Re-run with --go to copy."
