"""Shared typed result containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class OptimizationResult:
    method: str
    weights: np.ndarray
    objective: float
    evaluations: int
    runtime_seconds: float
    convergence_evaluations: list[int]
    convergence_best: list[float]
    success: bool = True
    message: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "weights": self.weights.tolist(),
            "objective": float(self.objective),
            "evaluations": int(self.evaluations),
            "runtime_seconds": float(self.runtime_seconds),
            "convergence_evaluations": [int(x) for x in self.convergence_evaluations],
            "convergence_best": [float(x) for x in self.convergence_best],
            "success": bool(self.success),
            "message": self.message,
            "diagnostics": self.diagnostics,
        }


@dataclass(frozen=True)
class WalkForwardWindow:
    index: int
    train_start: str
    train_end: str
    hold_start: str
    hold_end: str
    train_positions: np.ndarray
    hold_positions: np.ndarray
