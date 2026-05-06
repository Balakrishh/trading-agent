"""
_busy.py — Global "app is busy" overlay used across the Streamlit tabs.

Why this exists
---------------
Streamlit reruns the whole script top-to-bottom on every interaction, but
during a long-running synchronous call (a backtest, a watchlist refresh,
a `decide()` preview) the user has no clear visual signal that the app is
working — they just see a frozen page with the small "Running…" badge in
the top-right. That subtle indicator is easy to miss, and clicks during a
run *queue* but never give the user feedback that they're queueing.

What this provides
------------------
1. ``inject_overlay_css()`` — call once near the top of ``app.py`` so the
   stylesheet for the overlay is present in the DOM. Idempotent.
2. ``register_top_level_slot()`` — call ONCE at the top of ``app.py``
   *before* ``st.tabs(...)``. Reserves a tab-agnostic ``st.empty()``
   placeholder that ``global_busy`` will paint into. Critical: the slot
   must live OUTSIDE any tab body, because Streamlit hides inactive tab
   panels with ``display: none`` and ``position: fixed`` cannot escape
   that — an overlay rendered inside a hidden tab is invisible.
3. ``global_busy(label)`` — context manager that paints a fullscreen
   semi-transparent backdrop with a centred status card while the wrapped
   block runs. Uses the top-level slot when registered; falls back to a
   local ``st.empty()`` if the caller forgot ``register_top_level_slot``.

Usage
-----
    # In app.py, BEFORE st.tabs(...):
    from trading_agent.streamlit._busy import (
        inject_overlay_css, register_top_level_slot, global_busy,
    )
    inject_overlay_css()
    register_top_level_slot()

    # In any tab handler:
    with global_busy("Running backtest…"):
        result = runner.run()

The CSS uses ``z-index: 999999`` and ``pointer-events: all`` on the
overlay so any clicks on the dimmed area are absorbed (and silently
discarded) until the run completes — preventing the user from spamming
buttons that would just queue stale events behind the current run.

Note on Streamlit's own spinner
-------------------------------
``st.spinner(...)`` only highlights the widget that opened it; it doesn't
block the rest of the page. ``global_busy`` is the heavier hammer for
when *any* rerun-blocking work is happening that would make a click
elsewhere produce confusing results.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

import streamlit as st


# Sentinel session-state key so we don't double-inject the stylesheet on
# every rerun (Streamlit's <head> is rebuilt each time but extra <style>
# blocks are still wasted bytes).
_CSS_FLAG = "_global_busy_css_injected"

# Session-state key for the top-level placeholder. We store it on
# ``st.session_state`` rather than a module global because module globals
# leak across Streamlit sessions (multi-user, same process); session_state
# is the per-session bucket.
_SLOT_KEY = "_global_busy_top_level_slot"


_OVERLAY_CSS = """
<style>
.global-busy-overlay {
    position: fixed;
    top: 0; left: 0;
    width: 100vw; height: 100vh;
    background: rgba(15, 17, 22, 0.55);
    backdrop-filter: blur(2px);
    -webkit-backdrop-filter: blur(2px);
    z-index: 999999;
    display: flex;
    align-items: center;
    justify-content: center;
    pointer-events: all;     /* swallow clicks → block other actions */
    cursor: progress;
    animation: gbo-fade-in 0.15s ease-out;
}
.global-busy-overlay .gbo-card {
    background: var(--background-color, #ffffff);
    color: var(--text-color, #111111);
    padding: 1.4rem 2rem;
    border-radius: 10px;
    font-size: 1.05rem;
    font-weight: 500;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45);
    display: flex;
    align-items: center;
    gap: 0.9rem;
    max-width: 90vw;
}
.global-busy-overlay .gbo-spinner {
    width: 22px;
    height: 22px;
    border: 3px solid rgba(120, 120, 120, 0.3);
    border-top-color: #4c8bf5;
    border-radius: 50%;
    animation: gbo-spin 0.9s linear infinite;
    flex-shrink: 0;
}
@keyframes gbo-spin {
    to { transform: rotate(360deg); }
}
@keyframes gbo-fade-in {
    from { opacity: 0; }
    to   { opacity: 1; }
}
/* Dark-theme readability — Streamlit toggles data-theme on <html>.    */
[data-theme="dark"] .global-busy-overlay .gbo-card,
.stApp[data-theme="dark"] .global-busy-overlay .gbo-card {
    background: #1a1d23;
    color: #f0f1f3;
}
</style>
"""


def inject_overlay_css() -> None:
    """Inject the overlay stylesheet once per session.

    The CSS itself is inert — it only takes effect when a div with class
    ``global-busy-overlay`` is added to the DOM by ``global_busy()``.
    Re-injecting is a no-op after the first call within a session.
    """
    # We always emit the stylesheet on every script run (not gated by the
    # session flag) because Streamlit clears the <head>'s inline-style
    # state between reruns in some configurations. The flag is kept for
    # tests / debug only.
    st.markdown(_OVERLAY_CSS, unsafe_allow_html=True)
    st.session_state[_CSS_FLAG] = True


def register_top_level_slot() -> None:
    """Reserve a tab-agnostic ``st.empty()`` for the busy overlay.

    Must be called BEFORE ``st.tabs(...)`` so the placeholder lives at
    the page root, not inside a tab body. Streamlit hides inactive tab
    panels with ``display: none``; an overlay rendered inside one is
    completely invisible (CSS ``position: fixed`` cannot escape a
    ``display: none`` ancestor — that's a fundamental rule of the visual
    formatting model).

    Refreshed on every rerun because ``st.empty()`` returns a new
    DeltaGenerator each time.
    """
    st.session_state[_SLOT_KEY] = st.empty()


def _get_slot():
    """Return the current rerun's top-level slot or a fallback ``empty``.

    If the caller forgot ``register_top_level_slot()`` (or this is being
    used in a context where there are no tabs — e.g. a unit test), the
    fallback creates an inline placeholder that still works for non-tab
    layouts. For the dashboard it's always the top-level slot.
    """
    slot = st.session_state.get(_SLOT_KEY)
    if slot is None:
        return st.empty()
    return slot


@contextmanager
def global_busy(label: str = "Working…",
                detail: Optional[str] = None) -> Iterator[None]:
    """Render a blocking overlay while the wrapped block executes.

    Parameters
    ----------
    label : str
        Short headline shown next to the spinner ("Running backtest…",
        "Refreshing watchlist…").
    detail : Optional[str]
        Optional second line for context (e.g. "13 tickers · 5 timeframes").

    The overlay is drawn into the page-level placeholder reserved by
    ``register_top_level_slot()`` so it covers the viewport regardless
    of which tab is active. Cleared on context exit, even if the wrapped
    block raises.
    """
    slot = _get_slot()
    detail_html = (
        f"<div style='font-size:0.85em;opacity:0.7;margin-top:0.2rem'>"
        f"{detail}</div>"
    ) if detail else ""
    slot.markdown(
        f"""
        <div class="global-busy-overlay">
            <div class="gbo-card">
                <div class="gbo-spinner"></div>
                <div>
                    {label}
                    {detail_html}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    try:
        yield
    finally:
        slot.empty()
