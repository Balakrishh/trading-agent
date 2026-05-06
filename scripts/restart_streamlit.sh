#!/usr/bin/env bash
# restart_streamlit.sh — graceful Streamlit + agent-loop restart.
#
# What this does
# --------------
# 1. Finds anything listening on the Streamlit port (default 8501) and
#    sends it SIGTERM. The agent loop is a background thread inside the
#    Streamlit process, so killing Streamlit also stops the loop and any
#    in-flight cycle subprocess (which is parented to Streamlit).
# 2. Waits up to 10 seconds for graceful shutdown, then SIGKILLs anything
#    still alive on the port.
# 3. Removes the AGENT_RUNNING / AGENT_PID sentinel files so the next
#    Streamlit boot starts in a clean "stopped" state — the user explicitly
#    clicks Start in the Live tab to begin trading on the new config. This
#    matches the safety convention used by the dashboard's Stop button.
# 4. Re-launches Streamlit in the background with logs redirected to
#    logs/streamlit_restart.log.
#
# Why a script (not just `streamlit run` again)
# ---------------------------------------------
# - Picks up new TICKERS env var (Streamlit only reads .env at boot).
# - Picks up new STRATEGY_PRESET.json (also re-read per cycle, but a
#   restart guarantees the *whole* preset library cache is rebuilt).
# - Stops any orphaned cycle subprocess that survived a previous crash.
#
# Usage
# -----
#   ./scripts/restart_streamlit.sh           # default port 8501
#   STREAMLIT_PORT=8502 ./scripts/restart_streamlit.sh
#
# After restart, open http://localhost:${STREAMLIT_PORT:-8501} and click
# Start in the Live tab to resume trading on the new config.

set -euo pipefail

PORT="${STREAMLIT_PORT:-8501}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/streamlit_restart.log"

echo "[restart_streamlit] Repo root: ${REPO_ROOT}"
echo "[restart_streamlit] Port: ${PORT}"

# ── 1. Find and kill any process holding the port ────────────────────────
# `lsof -t -i :PORT` prints PIDs only. Robust on both macOS and Linux.
PIDS="$(lsof -t -i ":${PORT}" 2>/dev/null || true)"

if [ -n "${PIDS}" ]; then
    echo "[restart_streamlit] Killing PIDs on port ${PORT}: ${PIDS}"
    # SIGTERM first — give Streamlit a chance to flush logs / sentinels.
    kill ${PIDS} 2>/dev/null || true

    # Wait up to 10s for graceful shutdown.
    for i in $(seq 1 10); do
        sleep 1
        STILL_ALIVE="$(lsof -t -i ":${PORT}" 2>/dev/null || true)"
        if [ -z "${STILL_ALIVE}" ]; then
            echo "[restart_streamlit] Port ${PORT} freed after ${i}s"
            break
        fi
    done

    # If anything's still hanging on, SIGKILL it.
    STILL_ALIVE="$(lsof -t -i ":${PORT}" 2>/dev/null || true)"
    if [ -n "${STILL_ALIVE}" ]; then
        echo "[restart_streamlit] Force-killing stragglers: ${STILL_ALIVE}"
        kill -9 ${STILL_ALIVE} 2>/dev/null || true
        sleep 1
    fi
else
    echo "[restart_streamlit] Nothing listening on port ${PORT}"
fi

# ── 2. Clear agent sentinels so we boot in a clean "stopped" state ───────
# The Live tab's "Start" button writes these on the new boot — that's the
# explicit, user-visible re-entry. Leaving them stale would auto-resume
# trading immediately, which we don't want during a config change.
for sentinel in "AGENT_RUNNING" "AGENT_PID" "DRY_RUN" "PAUSE_FLAG"; do
    fp="${REPO_ROOT}/${sentinel}"
    if [ -e "${fp}" ]; then
        echo "[restart_streamlit] Removing stale sentinel: ${sentinel}"
        rm -f "${fp}"
    fi
done

# ── 3. Print the active config so the user can confirm what's loading ────
echo "[restart_streamlit] Active config:"
if [ -f "${REPO_ROOT}/.env" ]; then
    grep -E '^(TICKERS|MAX_DELTA|MIN_CREDIT_RATIO|MAX_RISK_PCT)=' \
         "${REPO_ROOT}/.env" | sed 's/^/    /'
fi
if [ -f "${REPO_ROOT}/STRATEGY_PRESET.json" ]; then
    echo "    STRATEGY_PRESET.json:"
    python3 -c "
import json, sys
with open('${REPO_ROOT}/STRATEGY_PRESET.json') as fp:
    p = json.load(fp)
c = p.get('custom', {})
print(f'      delta_grid={c.get(\"delta_grid\")}')
print(f'      width_grid_pct={c.get(\"width_grid_pct\")}')
print(f'      dte_grid={c.get(\"dte_grid\")}')
print(f'      edge_buffer={p.get(\"edge_buffer\")} max_delta={c.get(\"max_delta\")} '
      f'min_pop={c.get(\"min_pop\")} min_cw={c.get(\"min_credit_ratio\")}')
" 2>/dev/null || echo "      (could not parse)"
fi

# ── 4. Relaunch Streamlit ────────────────────────────────────────────────
cd "${REPO_ROOT}"

# Use nohup + & so the script returns to the user's shell. stdout/stderr
# go to LOG_FILE so the user can `tail -f logs/streamlit_restart.log` to
# watch boot progress. PID is captured for convenience.
echo "[restart_streamlit] Starting Streamlit on port ${PORT}…"
nohup streamlit run trading_agent/streamlit/app.py \
      --server.port "${PORT}" \
      --server.headless true \
      > "${LOG_FILE}" 2>&1 &

NEW_PID=$!
echo "[restart_streamlit] Streamlit PID: ${NEW_PID}"
echo "[restart_streamlit] Logs: ${LOG_FILE}"
echo
echo "[restart_streamlit] Wait ~3-5s for the page to be ready, then:"
echo "    open http://localhost:${PORT}"
echo
echo "Then: go to the Live tab → click Start to begin trading on the new config."
