# AI Show & Tell — Talking Points

**Slot:** 10-15 min · **Audience:** mixed technical + PM/business · **Format:** live demo, no slides · **Setup:** laptop + Streamlit + Telegram on your phone.

---

## The 30-second elevator (memorize this — say it first)

> "I've been co-developing an autonomous options-trading agent with Claude for about six months. It runs on a Raspberry Pi in my closet, trades paper money through Alpaca, and sends me Telegram alerts when it does something interesting. Today I want to show you two things: what the agent looks like when it's running, and how I've been shipping features into it — because the way an AI and I actually build software together turned out to be more interesting than the trading itself."

Deliver that with your laptop closed. Then open it and go to the dashboard.

---

## Time budget (15 min slot)

| Minute | What you're doing | Beat |
|---|---|---|
| 0-1 | Elevator pitch, laptop closed | Hook — "trades money in my closet" |
| 1-4 | Live Streamlit dashboard tour | Show it running |
| 4-7 | Long-Term Evaluator + Telegram demo | Show the AI collaboration payoff |
| 7-10 | SDD story — spec → conformance test → CI gate | Show HOW you build with AI |
| 10-13 | Reflection: what I learned | The takeaway that lands |
| 13-15 | Q&A buffer | Never overrun |

At 10 min elapsed, you should be transitioning to reflection. If you're behind, cut the CI-gate demo and go straight to reflection.

---

## Live demo runbook — click sequence

### Setup 30 seconds before you go on

Open on your laptop, in this order, one Chrome tab each:

1. Streamlit dashboard at `http://localhost:8501` — start on the Live Monitoring tab.
2. Your Telegram app or web.telegram.org, viewing the `long_term` channel.
3. A terminal window in `~/trading-agent`, activated venv, ready to run one command.
4. VS Code or GitHub with `docs/skills/40_long_term_options_evaluator.md` open.

Test each one works. If Streamlit is empty, refresh once. If Telegram isn't showing recent history, tap the channel first.

### Beat 1 — The Live Monitoring tab (2 min)

*What you say:* "This is the agent's live cockpit. It runs a 5-minute cycle: check volatility regime, monitor open positions, scan for new trades, place orders. Everything you see on the screen — the guardrail grid, the open positions table, the P&L — is derived from a single JSON event log. Every decision the agent makes writes a row; the dashboard is a read-only view of that log."

*What to point at:* the guardrail grid (say "each cell is a potential trade the agent considered this cycle"), the account balance widget, one recent journal row.

*Don't say:* specific credit-spread math, delta values, or "why credit spreads." Skip options theory entirely.

### Beat 2 — The Long-Term Evaluator tab (3 min)

Click the Long-Term Evaluator tab.

*What you say:* "This is what I built last month. The agent's main job is short-DTE credit spreads, but I also have longer-term stock positions in my brokerage account. This tab evaluates my holdings alongside my watchlist and suggests options moves — covered calls to earn income on stock I already own, cash-secured puts to enter tickers I've been eyeing. It's read-only by design; it never places an order, it just tells me what to look at."

Show the paste flow: the textarea is already populated (from your saved holdings). Point at the sector breakdown pie. Point at the one covered-call recommendation (NOK, probably).

*What to point at:* "This one covered-call suggestion right here — NOK, 100 shares, sell a call 45 days out at $15 strike, collect $18 credit. The tab shows me the entry legs, the take-profit target, and the stop-loss level. If I want to place it, I copy those numbers into my broker. The agent doesn't touch real money — that's the rule."

### Beat 3 — Trigger a live Telegram alert (2 min)

Switch to your terminal.

*What you say:* "Now here's the part I wanted to show you. The agent doesn't just render this in a browser — it hourly-digests everything into a Telegram channel so I get alerts on my phone during the trading day. Let me trigger one now."

Run:

```
python -m trading_agent.portfolio_alert_scheduler --force --print-body
```

While the command runs (~5 seconds), say: "The `--force` skips the market-hours check because we're demoing off-hours."

Switch to Telegram. Show the digest arriving in the `long_term` channel.

*What to point at:* the sector breakdown, the covered-call bracket sketch, the "watchlist entry candidates" section. Scroll if needed.

*What you say:* "This is the whole loop. My holdings → the analyzer → my phone. And this happens automatically every hour during market hours because it's folded into the agent's 5-minute cycle — no separate cron, no Cowork task, just a `_maybe_run_portfolio_review` step in the loop."

### Beat 4 — The SDD story (3 min)

Switch to VS Code. Open `docs/skills/40_long_term_options_evaluator.md`.

*What you say:* "OK — now the interesting part. This isn't a slideshow about how I use Claude. This is how Claude and I actually work together on this codebase. Every feature in this repo has a skill file — a spec document that describes what the thing does, the math, edge cases, and the tests that pin it down. I'll show you one."

Scroll to §2 (Mathematical Formula), pause on the covered-call score formula.

*What you say:* "Every time we build a new feature, we write the skill file first. Then Claude writes the code. Then we write conformance tests that assert the code still matches the skill."

Now scroll down in the same skill file to §4 (Edge Cases). Point at this bullet:

> **Strikes below cost basis silently filtered.** `_score_covered_call` returns `None` when `strike < cost_basis × 1.01`.

*What you say:* "This one line in the spec is a rule that costs me real money if it's wrong. If the agent lets me write a covered call at a strike below what I paid for the shares, I'd lock in a loss the moment I got assigned. So we wrote it into the spec. Then we wrote a test that pins the code to the spec."

Switch to `tests/conformance/test_skill_40_long_term_evaluator.py`. Show this test — the entire block is <15 lines and fits on one screen:

```python
def test_strike_below_cost_basis_rejects():
    """§4 — never write a CC strike below cost basis × 1.01."""
    out = _score_covered_call(
        short_call=_short_call(strike=199.0),   # < 200 × 1.01 = 202
        cost_basis=200.0,
        preset=_StubPreset(),
    )
    assert out is None
    verbose = _score_covered_call_with_reason(
        short_call=_short_call(strike=199.0),
        cost_basis=200.0,
        preset=_StubPreset(),
    )
    assert verbose["status"] == "rejected"
    assert verbose["reason"] == LT_REJECT_STRIKE_BELOW_COST_BASIS
```

*What to point at:* the two assertions. Read them out loud slowly: "The scorer returns nothing. The verbose scorer says rejected, with the reason 'strike below cost basis.'"

*What you say:* "The test docstring references section 4 of the skill file. If Claude ever produces code that lets me write a covered call at a losing strike — whether from a bug, a hallucination, or a well-meaning refactor — this test fails and CI blocks the PR. The spec says the rule. The test enforces the spec. And now the AI can't accidentally cost me money, because the guardrail lives in code, not in my vigilance."

Now switch to the terminal, run:

```
python scripts/checks/scan_invariant_check.py
```

Point at the output — four invariants, all green.

*What you say:* "This one is even stronger. Four architectural invariants scanned by AST walkers — they don't run the code, they parse the code and prove structural properties. One of them says the credit-to-width formula must appear identically in three specific files. Another says the scoring function can only be *defined* in two allowed places. If Claude tries to shadow-implement a scorer somewhere else, this check blocks the commit. This is the difference between 'the AI helped me write code' and 'the AI writes code inside a system I can trust' — the guardrails are code, not vigilance."

### Beat 5 — Reflection (3 min)

Close the laptop halfway (theatrical). Look at the room.

*What you say — memorize:*

> "Here's what I actually learned building this. When I started, I thought using an AI copilot would mean I could type less. What it actually turned into is this: I write more specifications, and I write them more precisely, than I ever did before. Because the specs are what keep the AI honest. The code volume is maybe 10× what I'd have written solo, but the *spec volume* went from zero to about forty documents — and those specs are the thing I could hand to another engineer tomorrow and they'd understand the system in a day."
>
> "The AI didn't replace me writing code. It replaced me writing code *carelessly*. Every shortcut it might take, we've encoded as a CI check that would catch it. And in return, I get an agent trading in my closet, sending me covered-call suggestions on Telegram, that I built in evenings and weekends."
>
> "I'm happy to go deeper on any of it in Q&A."

Stop. Don't fill silence.

---

## Q&A prep — likely questions

**"Is it making money?"**
> "It's paper trading — no real money at risk. On the credit-spread side, the results are inconclusive because the volatility regime for the last three months hasn't matched what the strategy is designed for; the agent's actually correctly *not trading* most days. That's a feature, not a bug — it has strict entry gates and won't force trades. The long-term evaluator tab is newer; it just started producing recommendations this month."

**"Why not just use ChatGPT / Copilot?"**
> "For code completion, they're great. But this system has ~2000 lines of production code, 40+ spec documents, and 8 CI gates that enforce architectural invariants. What I needed was an AI that could hold the whole thing in context and reason about consistency across files. That's why the spec-driven approach matters — the specs give the AI a map so it doesn't drift."

**"How do you know the AI didn't hallucinate a bug?"**
> "Two ways. First, the conformance tests — every skill file has a test that pins the documented behavior against the actual code. Second, the invariant scanner: if the AI suggests defining a scoring function in the wrong file, CI blocks the commit. The point isn't that the AI is perfect; it's that the failure modes are caught before they land."

**"Why Raspberry Pi?"**
> "Cost, mostly — the agent uses a couple hundred MB of RAM and needs to run 24/7. A Pi is $50 and draws two watts. Cloud would work too but this was cheaper and I already had one."

**"What broker are you using?"**
> "Alpaca for the credit-spread trades — paper account, $30k notional. Schwab for market data and options chains because their coverage of small caps and ADRs is materially better than Alpaca's free feed. Later this year I'll wire Schwab's Trader API for actual order placement, but that's a whole additional session of work — separate OAuth scope, real-money guardrails, a preview-confirm-place UI. Not shipping that until it's watertight."

**"How much did the AI actually write?"**
> "Rough guess: 80-90% of the code. But that number is misleading — I wrote 100% of the specs, and the specs are what made the code viable. The AI is very good at 'given this precise description, produce the code.' It's less reliable at 'figure out what I want.' So we split the work along that line."

**"What's the scariest thing about letting an AI touch production code?"**
> "Silent regressions. A test that passes but a behavior that changed. That's why I lean so hard on the invariant scanners — they're static checks that don't need me to think of every edge case. If the AI changes the credit-to-width floor formula in one file but not the other two, an AST walker catches it. If it defines a scoring helper in a third file, another walker catches it. Belt and suspenders."

**"Can I see the code?"**
> "Yeah, GitHub — I'll send the link. Fair warning, the CI gates are strict; PRs need to be spec-updated + traceability-regenerated + footers-restamped. It's a lot of process for a hobby project, but that's exactly what makes it AI-safe."

**"Are you worried about the AI making bad trading decisions?"**
> "The AI doesn't make trading decisions. The agent's decision logic is deterministic — score every candidate spread, apply gates, pick the highest EV. The AI helped me *write* that logic. Every decision the agent makes writes a row to the journal, so I can audit exactly why it did what it did after the fact. There's no black box."

**"What would you build next?"**
> "Two things. First, wire the Schwab Trader API for real order placement — with a preview-confirm-place flow so the operator (me) always taps the button, not the AI. Second, add the cash-secured put and LEAPS scorers so the watchlist tickers I don't own yet also get concrete entry suggestions in the hourly digest. Both are 'next session' work — the specs are already drafted."

**"How do you avoid the AI just agreeing with you?"**
> "You have to argue with it. When I make a bad architectural suggestion, Claude will build it, but it'll also push back if the change violates a documented invariant. The specs make that possible — without them, the AI has no basis to disagree because 'the code' can mean anything."

---

## Contingency plans

**Streamlit crashes or is slow.** Skip Beat 1, go straight to the Long-Term Evaluator tab from your bookmarks. If that also fails, jump to Beat 3 — just run the CLI. Nothing depends on the Streamlit dashboard actually rendering.

**Telegram doesn't deliver.** Run the CLI with `--print-body` — the digest body prints to stdout. Screen-share the terminal. Say: "This is the exact body that would post to Telegram; the send is idempotent so I can trigger it any time."

**Terminal command fails.** Have a screenshot of a recent digest on your phone. Say "here's what one looked like this morning" and move on. Do not debug in front of the audience.

**Someone asks a technical question you don't know.** "Great question, I'd have to check the code to give you a precise answer. Grab me after."

**Someone asks about real money.** "This is paper trading. Real-money order placement is a separate session I haven't shipped yet. When it lands, the operator — me — has to click a button; the AI never places orders on my behalf."

---

## The one line to end on

If you have to cut everything at the end, land this:

> "The AI didn't replace me writing code. It replaced me writing code carelessly. And the specs are the thing that made that possible."

That's the takeaway. Everything else is texture.
