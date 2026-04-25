#!/bin/bash
# ============================================================
# Intelligence Layer Setup
# ============================================================
# This script installs and configures the local LLMs (Ollama)
# that power the trading agent's tiered intelligence layer:
#
#   • Analyst        — general trade reasoning       (mistral / llama3 / qwen2.5)
#   • Embeddings     — RAG knowledge base            (nomic-embed / mxbai)
#   • FinGPT         — finance-tuned sentiment       (qwen2.5 7B / 32B)
#   • Verifier       — anti-hallucination CoT guard  (qwq:32b / deepseek-r1)
#
# Two axes of configuration:
#
#   TIER     — which roles to install
#     minimal   → analyst + embeddings                (default)
#     sentiment → + FinGPT specialist
#     full      → + reasoning verifier
#
#   PROFILE  — size class for the four models
#     standard  → 7-32B stack, ~5-25 GB RAM per model  (default)
#     high-mem  → 32-70B stack, for 64GB+ unified memory hosts
#                 (tuned for Apple Silicon Pro/Max with 64-128GB RAM
#                 and 20+ GB/s Metal bandwidth — see notes below)
#
# Per-model overrides (always win over the profile default):
#   ANALYST_MODEL   EMBED_MODEL   FINGPT_MODEL   VERIFIER_MODEL
#
# Cloud verifier users (Anthropic) should leave TIER=sentiment and
# instead set VERIFIER_PROVIDER=anthropic + VERIFIER_API_KEY in .env
# — no local pull is required for that path.
#
# Prerequisites: Ollama must be installed (https://ollama.com)
# ============================================================

set -e

TIER="${TIER:-minimal}"
PROFILE="${PROFILE:-standard}"

# ------------------------------------------------------------
# Resolve per-profile defaults.  Explicit overrides — i.e. a user
# who exported FINGPT_MODEL=... before running the script — still
# win because `${VAR:=default}` only fires when VAR is unset.
# ------------------------------------------------------------
case "$PROFILE" in
    standard)
        : "${ANALYST_MODEL:=mistral}"
        : "${EMBED_MODEL:=nomic-embed-text}"
        : "${FINGPT_MODEL:=qwen2.5:7b}"
        : "${VERIFIER_MODEL:=qwq:32b}"
        ;;
    high-mem)
        # Recommended stack for 64-128GB unified-memory Macs.
        # Footprint at Q4_K_M (Ollama's default quantization):
        #   Analyst  70B  ≈ 40 GB
        #   FinGPT   32B  ≈ 19 GB
        #   Verifier 70B  ≈ 40 GB
        #   Embed          ≈  1 GB
        #   Total          ≈ 100 GB resident if all kept warm.
        : "${ANALYST_MODEL:=llama3.1:70b}"
        : "${EMBED_MODEL:=mxbai-embed-large}"
        : "${FINGPT_MODEL:=qwen2.5:32b-instruct}"
        : "${VERIFIER_MODEL:=deepseek-r1:70b}"
        ;;
    *)
        echo "ERROR: Unknown PROFILE='$PROFILE'. Use: standard | high-mem"
        exit 2
        ;;
esac

case "$TIER" in
    minimal|sentiment|full) ;;
    *)
        echo "ERROR: Unknown TIER='$TIER'. Use one of: minimal | sentiment | full"
        exit 2
        ;;
esac

echo "============================================"
echo "Trading Agent — Intelligence Layer Setup"
echo "============================================"
echo "  Profile     : $PROFILE"
echo "  Tier        : $TIER"
echo "  Analyst     : $ANALYST_MODEL"
echo "  Embeddings  : $EMBED_MODEL"
if [ "$TIER" != "minimal" ]; then
    echo "  FinGPT      : $FINGPT_MODEL"
fi
if [ "$TIER" = "full" ]; then
    echo "  Verifier    : $VERIFIER_MODEL"
fi
if [ "$PROFILE" = "high-mem" ]; then
    echo ""
    echo "  [high-mem]  Expect ~40-100 GB of downloads depending on TIER."
    echo "              Ensure ≥ 64 GB unified memory and ≥ 150 GB free SSD."
fi
echo "============================================"
echo ""

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "ERROR: Ollama is not installed."
    echo ""
    echo "Install Ollama:"
    echo "  macOS/Linux: curl -fsSL https://ollama.com/install.sh | sh"
    echo "  Or download from: https://ollama.com/download"
    echo ""
    exit 1
fi

# ============================================================
# Helper — idempotent pull
# ============================================================
# Skip the network round-trip when the model is already cached.
# `ollama list` prints "<model>:<tag>   <id>   <size>   <age>"; we
# match the tag the user actually requested.
# ============================================================
pull_if_missing() {
    local model="$1"
    local label="$2"
    if ollama list 2>/dev/null | awk '{print $1}' | grep -qx "$model"; then
        echo "  ✓ $label already installed ($model) — skipping pull"
    else
        echo "  Pulling $label ($model)..."
        ollama pull "$model"
    fi
}

# ============================================================
# Helper — chat smoke-test
# ============================================================
test_chat_model() {
    local model="$1"
    local label="$2"
    echo "Testing $label ($model)..."
    local response
    response=$(curl -s http://localhost:11434/api/chat -d "{
      \"model\": \"$model\",
      \"messages\": [{\"role\": \"user\", \"content\": \"Reply with exactly: OK\"}],
      \"stream\": false
    }" | python3 -c "import sys,json; print(json.load(sys.stdin).get('message',{}).get('content','FAIL'))" 2>/dev/null || echo "FAIL")

    if echo "$response" | grep -qi "ok"; then
        echo "  ✓ $label responds"
    else
        echo "  ⚠ $label test returned unexpected output: $response"
    fi
}

echo "[1/5] Checking Ollama service..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "  ✓ Ollama is running"
else
    echo "  Starting Ollama..."
    ollama serve &
    sleep 3
fi

# ============================================================
# Model Selection Guide
# ============================================================
#
# PROFILE=standard  (≤ 32 GB RAM hosts, most laptops)
# ------------------------------------------------------------
#   Analyst   : mistral 7B          ~4 GB     [profile default]
#   Embed     : nomic-embed-text    ~0.5 GB   [profile default]
#   FinGPT    : qwen2.5:7b          ~5 GB     [profile default]
#   Verifier  : qwq:32b             ~20 GB    [profile default]
#   Lighter alternates: phi3 (analyst), deepseek-r1:7b (verifier).
#
# PROFILE=high-mem  (64-128 GB unified-memory Macs, RTX 6000, etc.)
# ------------------------------------------------------------
#   Analyst   : llama3.1:70b        ~40 GB    [profile default]
#   Embed     : mxbai-embed-large   ~0.7 GB   [profile default]
#   FinGPT    : qwen2.5:32b-instruct ~19 GB   [profile default]
#   Verifier  : deepseek-r1:70b     ~40 GB    [profile default]
#   Stronger alternates:
#     qwen2.5:72b-instruct          (analyst — finance-adjacent edge)
#     qwq:32b                       (verifier — faster, slightly lower recall)
#     deepseek-r1:70b-q8_0          (verifier — Q8 if only this one runs)
#
# REASONING ON ROLES (do not cross-wire these!)
# ------------------------------------------------------------
#   • Analyst reasons over the TRADE (regime + sentiment + history).
#   • FinGPT scores news/filings TEXT only — not regime, not strikes.
#   • Verifier AUDITS FinGPT's claim against the source text.
#   • Strike selection, POP, sizing, order submission remain
#     deterministic Python — never an LLM responsibility.
#
# Cloud alternative for the verifier:
#   Set VERIFIER_PROVIDER=anthropic in .env (skips the pull entirely).
#
# FOR FINE-TUNING (after accumulating 50+ trades):
#   Use unsloth or Ollama's Modelfile for LoRA fine-tuning
#   Export training data with: python -m trading_agent.fine_tuning
#
# TAG VERIFICATION:
#   Ollama registry tags shift over time.  Before a large pull, run
#     ollama show <tag>
#   to confirm the variant exists and has the quantization you expect.
# ============================================================

echo ""
echo "[2/5] Analyst model..."
pull_if_missing "$ANALYST_MODEL" "analyst"

echo ""
echo "[3/5] Embedding model..."
pull_if_missing "$EMBED_MODEL" "embeddings"

if [ "$TIER" != "minimal" ]; then
    echo ""
    echo "[4/5] FinGPT specialist..."
    pull_if_missing "$FINGPT_MODEL" "FinGPT specialist"
else
    echo ""
    echo "[4/5] FinGPT specialist — skipped (TIER=minimal)."
fi

if [ "$TIER" = "full" ]; then
    echo ""
    echo "[5/5] Reasoning verifier..."
    pull_if_missing "$VERIFIER_MODEL" "verifier"
else
    echo ""
    echo "[5/5] Reasoning verifier — skipped (TIER=$TIER)."
    echo "      Cloud verifier? Set VERIFIER_PROVIDER=anthropic in .env."
fi

echo ""
echo "============================================"
echo "Verifying installed models"
echo "============================================"
echo ""
echo "Available models:"
ollama list
echo ""

# ------------------------------------------------------------
# Smoke tests — each tier exercises only the models it pulled.
# ------------------------------------------------------------
test_chat_model "$ANALYST_MODEL" "analyst"

echo "Testing embedding model ($EMBED_MODEL)..."
EMBED_RESULT=$(curl -s http://localhost:11434/api/embed -d "{
  \"model\": \"$EMBED_MODEL\",
  \"input\": [\"test\"]
}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('embeddings',[[]])[0]))" 2>/dev/null || echo "0")

if [ "$EMBED_RESULT" -gt 0 ] 2>/dev/null; then
    echo "  ✓ Embedding model works (${EMBED_RESULT}-dimensional vectors)"
else
    echo "  ⚠ Embedding test returned unexpected result"
fi

if [ "$TIER" != "minimal" ]; then
    test_chat_model "$FINGPT_MODEL" "FinGPT specialist"
fi

if [ "$TIER" = "full" ]; then
    test_chat_model "$VERIFIER_MODEL" "verifier"
fi

# ------------------------------------------------------------
# Final operator guidance
# ------------------------------------------------------------
echo ""
echo "============================================"
echo "Setup complete!"
echo "============================================"
echo ""
echo "To enable the intelligence layer, edit .env:"
echo ""
echo "  # Core analyst + RAG"
echo "  LLM_ENABLED=true"
echo "  LLM_MODEL=$ANALYST_MODEL"
echo "  EMBEDDING_MODEL=$EMBED_MODEL"
echo ""
if [ "$TIER" != "minimal" ]; then
    echo "  # Tier-2: FinGPT specialist (news/filings sentiment)"
    echo "  FINGPT_ENABLED=true"
    echo "  FINGPT_MODEL=$FINGPT_MODEL"
    echo ""
fi
if [ "$TIER" = "full" ]; then
    echo "  # Tier-2: reasoning verifier (anti-hallucination)"
    echo "  VERIFIER_ENABLED=true"
    echo "  VERIFIER_PROVIDER=ollama"
    echo "  VERIFIER_MODEL=$VERIFIER_MODEL"
    echo ""
else
    echo "  # Cloud verifier alternative (no local pull required):"
    echo "  #   VERIFIER_ENABLED=true"
    echo "  #   VERIFIER_PROVIDER=anthropic"
    echo "  #   VERIFIER_MODEL=claude-sonnet-4-6"
    echo "  #   VERIFIER_API_KEY=sk-ant-..."
    echo ""
fi
echo "  # Tier-0: earnings short-circuit (yfinance, no API key)"
echo "  EARNINGS_CALENDAR_ENABLED=true"
echo "  EARNINGS_LOOKAHEAD_DAYS=7"
echo ""
echo "  # Tier-1: content-hash sentiment cache"
echo "  NEWS_CACHE_TTL=900            # seconds; 15 min default"
echo "  SENTIMENT_HASH_CACHE_SIZE=512"
echo ""
echo "Then run the agent:"
echo "  python -m trading_agent.agent"
echo ""
echo "The agent will now:"
echo "  • Short-circuit on Tier-0 earnings windows (no LLM spend)"
echo "  • Replay Tier-1 cached sentiment on duplicate news"
echo "  • Run the Tier-2 NewsAggregator → FinGPT → Verifier chain"
echo "    only on novel evidence, inside a cycle-scoped pool"
echo "  • Analyze trades with LLM reasoning"
echo "  • Learn from past trade outcomes (RAG)"
echo "  • Recommend parameter adjustments over time"
echo "  • Export training data for fine-tuning"
echo ""
echo "Re-run with a bigger tier any time, e.g.:"
echo "  TIER=sentiment ./setup_intelligence.sh"
echo "  TIER=full      ./setup_intelligence.sh"
echo ""
echo "For 64-128 GB unified-memory hosts, upgrade the whole stack with:"
echo "  PROFILE=high-mem TIER=full ./setup_intelligence.sh"
echo ""
echo "Override individual models, e.g.:"
echo "  TIER=full VERIFIER_MODEL=deepseek-r1:7b ./setup_intelligence.sh"
echo "  PROFILE=high-mem ANALYST_MODEL=qwen2.5:72b-instruct ./setup_intelligence.sh"
echo "============================================"
