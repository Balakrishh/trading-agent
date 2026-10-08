# Runbook 08 — Move the agent to a Raspberry Pi

**Goal:** run the agent, the daily reviewer and (optionally) the dashboard on an always-on Raspberry Pi instead of the Mac, carrying over every journal, plan, preset and secret, and continue working with Claude Code on the Pi with the same project memory and conversation history. Backlog §9 ("always-on host").

**Time:** about 1 hour, most of it installing. **When:** the cutover (§6–§7) must happen **outside market hours** — after 16:20 ET or at a weekend.

**Assumes:** Pi user `balakrishh`, hostname `myrasberrypi.local` (as in runbook 05), repo at `/home/balakrishh/Documents/trading-agent`. Change `PI_HOST` / `PI_REPO` if yours differ.

> **Never run two agents.** The Mac and the Pi use the same Alpaca paper account: two running agents would each open trades. They also share one Schwab login whose refresh token rotates on every use, so whichever machine refreshes second is locked out. Stop everything on the Mac before starting the Pi (§6.1).

---

## 1. Hardware

| Item | Why |
|---|---|
| Raspberry Pi 4 or 5, **4 GB RAM or more** | the agent, the dashboard and Claude Code together use about 1–1.5 GB |
| **USB SSD** (Pi 4/5) or NVMe HAT (Pi 5) as the boot disk | the agent writes journal and state files every ~75 s; SD cards wear out and corrupt |
| Official power supply (Pi 5: 27 W, Pi 4: 15 W) | weak supplies cause random reboots |
| Ethernet cable to the router | steadier than Wi-Fi |
| Heatsink + fan; any mounting orientation; a few cm of air around it | the Pi slows itself down above ~80 °C |
| Small UPS (optional) | rides through short power cuts |

## 2. Operating system

1. Flash **Raspberry Pi OS Lite (64-bit)** to the SSD with Raspberry Pi Imager. In Imager's settings: hostname `myrasberrypi`, user `balakrishh`, enable SSH with your Mac's public key, timezone `America/New_York`.
2. Boot from the SSD and log in from the Mac: `ssh balakrishh@myrasberrypi.local`
3. Update and install the build tools:

   ```bash
   sudo apt update && sudo apt full-upgrade -y
   sudo apt install -y git python3-venv python3-dev build-essential rsync tmux
   sudo timedatectl set-timezone America/New_York   # the journal's day boundary uses local time, as on the Mac
   python3 --version                                # 3.11 (Bookworm) or 3.13 (Trixie) — both are tested in CI
   ```

## 3. Code

Code on the Mac is fully pushed: `main` = `origin/main`. What is *not* in git — journals, plans, `.env`, the Schwab token, Claude history — is copied in §6.

If the June copy is still on the Pi, move it aside (its journals are stale and would mix with the real ones):

```bash
[ -d ~/Documents/trading-agent ] && mv ~/Documents/trading-agent ~/Documents/trading-agent-june
```

Clone and install:

```bash
mkdir -p ~/Documents && cd ~/Documents
git clone https://github.com/Balakrishh/trading-agent.git
cd trading-agent
python3 -m venv myenv
myenv/bin/pip install --upgrade pip
myenv/bin/pip install -r requirements.txt        # numpy / pandas / scipy have ARM64 wheels; ~5–10 min
myenv/bin/python -m pytest tests/ -q             # expect all passed
myenv/bin/python scripts/checks/scan_invariant_check.py
```

## 4. Remote Git access (optional)

To `git push` from the Pi (for example when Claude Code commits there), add the Pi's SSH key to GitHub: `ssh-keygen -t ed25519`, paste `~/.ssh/id_ed25519.pub` into GitHub → Settings → SSH keys, then `git remote set-url origin git@github.com:Balakrishh/trading-agent.git`.

## 5. Install the services (do not start them yet)

systemd *user* services replace the Mac's launchd jobs:

| Mac (launchd) | Pi (systemd) |
|---|---|
| `com.trading-agent.headless` | `trading-agent.service` |
| `com.trading-agent.daily-reviewer` (16:15 weekdays) | `trading-agent-reviewer.timer` → `trading-agent-reviewer.service` |
| `scripts/restart_streamlit.sh` | `trading-agent-dashboard.service` (optional) |

```bash
mkdir -p ~/.config/systemd/user
cp ~/Documents/trading-agent/deploy/systemd/*.service ~/Documents/trading-agent/deploy/systemd/*.timer ~/.config/systemd/user/
systemctl --user daemon-reload
sudo loginctl enable-linger balakrishh            # run the services without anyone logged in, and at boot
```

Logs go to the same places as on the Mac: `logs/trading_agent.log` in the repo, plus `/tmp/trading-agent.headless.{out,err}.log` and `/tmp/trading-agent.daily-reviewer.{out,err}.log`.

## 6. Cutover — on the Mac, outside market hours

### 6.1 Stop everything on the Mac

```bash
launchctl bootout gui/$(id -u)/com.trading-agent.headless
launchctl bootout gui/$(id -u)/com.trading-agent.daily-reviewer
pkill -f 'streamlit run' || true
# Keep them from coming back at the next login:
mkdir -p ~/trading-agent-launchd-backup
mv ~/Library/LaunchAgents/com.trading-agent.*.plist ~/trading-agent-launchd-backup/
pgrep -fl trading_agent || echo "Mac is clear"
```

Quit any Claude Code session open in this repo on the Mac (its MCP server can refresh the Schwab token).

### 6.2 Copy state, secrets and Claude context

```bash
cd ~/trading-agent
scripts/migrate_to_pi.sh          # dry run: shows what will copy
scripts/migrate_to_pi.sh --go
```

It copies, over SSH:

| What | Where on the Pi |
|---|---|
| `trade_journal/`, `trade_plans/`, `daily_reviews/`, `knowledge_base/`, `pending_orders/`, `pending_preset_updates/`, `journal_kb/`, `logs/`, `STRATEGY_PRESET.json`, `AGENT_LOG` | repo |
| `.env` (Alpaca, Schwab, Telegram keys) | repo |
| `.claude/settings.local.json`, `docs/plans/journal_learnings.md` (local edits) | repo |
| `~/.schwab_tokens.json` (mode 600) | home |
| Claude Code project memory and conversation history | `~/.claude/projects/-home-balakrishh-Documents-trading-agent/` |

## 7. Start on the Pi

```bash
cd ~/Documents/trading-agent
myenv/bin/python -m trading_agent.schwab_oauth status       # token valid? if not: … schwab_oauth login (§9)
myenv/bin/python -m trading_agent.trading_halt status       # kill switch state carried over
systemctl --user enable --now trading-agent.service trading-agent-reviewer.timer
systemctl --user enable --now trading-agent-dashboard.service   # optional: http://myrasberrypi.local:8501
systemctl --user status trading-agent.service
tail -f /tmp/trading-agent.headless.err.log                 # outside hours: "Sleeping N seconds until next NYSE open"
systemctl --user list-timers | grep reviewer                 # next run: 16:15 ET on the next weekday
```

At the next open, check the first cycles: `tail -f logs/trading_agent.log` should show `Account: balance=…`, the six open positions in the monitor stage, and `TRADING CYCLE COMPLETE`.

## 8. Claude Code on the Pi

```bash
curl -fsSL https://claude.ai/install.sh | bash               # native Linux ARM64 build
tmux new -s claude                                            # survives SSH disconnects; reattach: tmux attach -t claude
cd ~/Documents/trading-agent && source myenv/bin/activate     # .mcp.json runs "python" → must be the venv's
claude                                                        # first run: /login prints a URL — open it on the Mac
```

- **Memory** (`MEMORY.md` and its files) loads automatically: it was copied into the Pi's project folder.
- **This conversation:** `claude --resume` and pick it from the list (or `claude --continue` for the most recent). Older messages mention Mac paths (`/Users/…`); the repo is now at `~/Documents/trading-agent`.
- **MCP:** run `/mcp` and check that `trading-agent` is connected; `/portfolio` and `/review` work as before.
- Not available on the Pi: Claude in Chrome (no browser). Everything else in the repo workflow is the same.

## 9. Day-to-day on the Pi

| Task | Command |
|---|---|
| Status / restart / stop the agent | `systemctl --user status|restart|stop trading-agent` (restart only outside market hours) |
| Pause / resume new entries | `myenv/bin/python -m trading_agent.trading_halt pause --reason "…"` / `resume` |
| Deploy new code | `git pull && systemctl --user restart trading-agent` (after the close) |
| Logs | `tail -f logs/trading_agent.log`; `journalctl --user -u trading-agent` |
| **Schwab re-login — every 7 days** | `myenv/bin/python -m trading_agent.schwab_oauth login`: open the printed URL in the Mac's browser, approve, paste the redirected `https://127.0.0.1:8182/…` URL back into the SSH session |
| Temperature / throttling | `vcgencmd measure_temp` (want < 70 °C); `vcgencmd get_throttled` (want `0x0`) |
| Disk | `df -h /`; the journal grows ~20 MB a week |

**Backups.** The journals now live only on the Pi. From the Mac, pull a copy weekly:
`rsync -az balakrishh@myrasberrypi.local:Documents/trading-agent/{trade_journal,trade_plans,daily_reviews,STRATEGY_PRESET.json} ~/trading-agent-pi-backup/`

## 10. Roll back to the Mac

1. On the Pi: `systemctl --user disable --now trading-agent.service trading-agent-reviewer.timer trading-agent-dashboard.service`
2. On the Mac, copy state back:
   `rsync -az balakrishh@myrasberrypi.local:Documents/trading-agent/{trade_journal,trade_plans,daily_reviews,pending_orders,STRATEGY_PRESET.json} ~/trading-agent/` and `rsync -az balakrishh@myrasberrypi.local:.schwab_tokens.json ~/`
3. `mv ~/trading-agent-launchd-backup/*.plist ~/Library/LaunchAgents/` and `launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.trading-agent.headless.plist` (same for the reviewer).

## 11. Edge cases

- **Power cut or reboot:** linger + `WantedBy=default.target` start the agent at boot; the supervisor sleeps until the next open. A missed open still goes unnoticed until the liveness alert (backlog §9) exists.
- **Clock:** the Pi has no battery-backed clock; it syncs over the network at boot. The supervisor uses the NYSE calendar and wall-clock time, so a wrong clock before sync only delays the first cycle.
- **Schwab token expired** (after 7 days without a re-login): quote calls fail and cycles error. Re-login (§9); nothing else needs restarting.
- **Python 3.11 on Bookworm vs 3.14 on the Mac:** CI tests 3.11, 3.12 and 3.14.

---

*Last verified against repo HEAD on 2026-10-07.*
