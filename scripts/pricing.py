"""Per-token API pricing, USD per 1M tokens, standard tier / short context.

Sourced from developers.openai.com/api/docs/pricing and the per-model pages
(fetched 2026-08-31). Prices change -- re-check before quoting any cost figure in
a writeup. `None` means the rate is unknown/unverified rather than zero, and the
cost helpers will refuse to produce a number rather than silently undercount.

Reasoning tokens are billed at the OUTPUT rate and are already included in the
OpenAI `completion_tokens` total, so no separate line item is needed.
"""

# model id -> (input, cached_input, output)
OPENAI_PRICING = {
    "gpt-5.6-sol": (4.00, 0.40, 20.00),
    "gpt-5.6-terra": (2.00, 0.20, 12.00),
    "gpt-5.6-luna": (0.20, 0.02, 1.20),
    "gpt-5.5": (5.00, 0.50, 30.00),
    "gpt-5.4": (2.50, 0.25, 15.00),
    "gpt-5.4-mini": (0.75, 0.075, 4.50),
    "gpt-5.4-nano": (0.20, 0.02, 1.25),
    "gpt-5-mini": (0.25, 0.025, 2.00),
    "gpt-5-nano": (0.05, 0.005, 0.40),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-4o-mini": (0.15, 0.075, 0.60),
}

# From platform.claude.com/docs/en/about-claude/pricing (fetched 2026-08-31). "Cached
# input" here is Anthropic's cache-hit rate; cache *writes* cost 1.25x base input and
# are tracked separately in the run output.
ANTHROPIC_PRICING = {
    "claude-opus-5": (5.00, 0.50, 25.00),
    "claude-sonnet-5": (2.00, 0.20, 10.00),
    "claude-haiku-4-5": (1.00, 0.10, 5.00),
}

PRICING = {"openai": OPENAI_PRICING, "anthropic": ANTHROPIC_PRICING}


def rates(provider, model):
    """Return (input, cached_input, output) USD per 1M tokens, or None if unknown."""
    table = PRICING.get(provider, {})
    if model in table:
        return table[model]
    # Dated snapshots (gpt-5.4-mini-2026-03-17) bill at their alias's rate.
    for name, r in table.items():
        if model.startswith(name + "-"):
            return r
    return None


def cost_usd(provider, model, uncached_input, cached_input, output):
    """Total USD for one run, or None if this model's rates aren't known."""
    r = rates(provider, model)
    if r is None:
        return None
    rate_in, rate_cached, rate_out = r
    return (
        uncached_input * rate_in / 1e6
        + cached_input * rate_cached / 1e6
        + output * rate_out / 1e6
    )
