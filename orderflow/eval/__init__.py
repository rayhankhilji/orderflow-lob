"""Benchmark + selection harness (the final arbiter for agent branches)."""

from orderflow.eval import adversarial, execution_bench, prediction_bench, select  # noqa: F401
from orderflow.eval.execution_bench import BenchConfig, run_bench, run_episode  # noqa: F401
from orderflow.eval.prediction_bench import PredBenchConfig, run_pred_bench  # noqa: F401
from orderflow.eval.select import (  # noqa: F401
    exec_scores,
    pred_scores,
    write_leaderboard,
)
