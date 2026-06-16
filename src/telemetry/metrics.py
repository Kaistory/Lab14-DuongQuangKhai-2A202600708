import time
from typing import Dict, Any, List
from src.telemetry.logger import logger

# Real per-1K-token pricing (USD), split input/output, matched by model substring.
# Source: public OpenAI / Google pricing for the model versions this lab uses.
# Local models run on the student's own CPU/GPU -> no marginal API cost.
_PRICING: Dict[str, Dict[str, float]] = {
    "gpt-4o-mini": {"in": 0.00015, "out": 0.00060},
    "gpt-4o":      {"in": 0.00250, "out": 0.01000},
    "gpt-4-turbo": {"in": 0.01000, "out": 0.03000},
    "gpt-3.5":     {"in": 0.00050, "out": 0.00150},
    "gemini-3.1-flash-lite": {"in": 0.00010, "out": 0.00040},  # ước lượng tier flash-lite
    "gemini-2.5-flash":      {"in": 0.000075, "out": 0.00030},
    "gemini-1.5-flash":      {"in": 0.000075, "out": 0.00030},
    "gemini-1.5-pro":        {"in": 0.00125,  "out": 0.00500},
    "gemini-2.0-flash":      {"in": 0.000075, "out": 0.00030},
}
# Models priced at $0 (local inference / test mocks).
_FREE_HINTS = ("local", "mock", "phi", "llama", "qwen", "mistral", "gguf")


class PerformanceTracker:
    """
    Tracking industry-standard metrics for LLMs.
    """
    def __init__(self):
        self.session_metrics = []

    def track_request(self, provider: str, model: str, usage: Dict[str, int], latency_ms: int):
        """
        Logs a single request metric to our telemetry.
        """
        metric = {
            "provider": provider,
            "model": model,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "latency_ms": latency_ms,
            "cost_estimate": self._calculate_cost(model, usage) # Mock cost calculation
        }
        self.session_metrics.append(metric)
        logger.log_event("LLM_METRIC", metric)

    def _calculate_cost(self, model: str, usage: Dict[str, int]) -> float:
        """
        Estimate request cost from real per-model pricing, billing prompt and
        completion tokens at their different rates (industry-accurate).

        Falls back to a conservative gpt-4o-mini-like rate for unknown hosted
        models, and to $0 for local / mock inference.
        """
        name = (model or "").lower()
        if any(h in name for h in _FREE_HINTS):
            return 0.0

        rate = None
        for key, r in _PRICING.items():
            if key in name:
                rate = r
                break
        if rate is None:
            rate = {"in": 0.00015, "out": 0.00060}  # default: cheap hosted tier

        prompt = usage.get("prompt_tokens", 0)
        completion = usage.get("completion_tokens", 0)
        if not prompt and not completion:
            # Some providers only report a total -> price it all at the input rate.
            prompt = usage.get("total_tokens", 0)
        cost = (prompt / 1000.0) * rate["in"] + (completion / 1000.0) * rate["out"]
        return round(cost, 8)

# Global tracker instance
tracker = PerformanceTracker()
