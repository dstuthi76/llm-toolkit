"""Track $ spent and $ saved vs. an "always use the expensive model" baseline."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class LogEntry:
    prompt_preview: str
    model_used: str
    escalated: bool
    actual_cost: float
    baseline_cost: float  # what it would have cost on the expensive model


@dataclass
class CostTracker:
    entries: list[LogEntry] = field(default_factory=list)

    def log(self, entry: LogEntry) -> None:
        self.entries.append(entry)

    @property
    def total_actual_cost(self) -> float:
        return sum(e.actual_cost for e in self.entries)

    @property
    def total_baseline_cost(self) -> float:
        return sum(e.baseline_cost for e in self.entries)

    @property
    def total_saved(self) -> float:
        return self.total_baseline_cost - self.total_actual_cost

    @property
    def escalation_rate(self) -> float:
        if not self.entries:
            return 0.0
        return sum(1 for e in self.entries if e.escalated) / len(self.entries)

    def summary(self) -> str:
        n = len(self.entries)
        return (
            f"{n} requests | escalation rate: {self.escalation_rate:.0%} | "
            f"spent: ${self.total_actual_cost:.4f} | "
            f"would have spent (always-expensive baseline): ${self.total_baseline_cost:.4f} | "
            f"saved: ${self.total_saved:.4f} ({self._pct_saved():.0%})"
        )

    def _pct_saved(self) -> float:
        if self.total_baseline_cost == 0:
            return 0.0
        return self.total_saved / self.total_baseline_cost
