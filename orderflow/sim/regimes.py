"""Named parameter presets: calm, normal, volatile, thin.

Each regime maps to (flow kwargs, seed_book kwargs) for every flow model.
Targets for ``normal`` at tick_size=0.01, mid ~ $100: ~5-15 events/sec,
spread mostly 1-2 ticks, ~1-3 bps/min realised vol. ``volatile`` ~3x vol,
``thin`` ~1/4 depth, ``calm`` slower/tighter.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from orderflow.sim.flow.hawkes import HawkesFlow
from orderflow.sim.flow.queue_reactive import QueueReactiveFlow
from orderflow.sim.flow.zero_intelligence import ZeroIntelligenceFlow

FLOWS = {
    "zi": ZeroIntelligenceFlow,
    "zero_intelligence": ZeroIntelligenceFlow,
    "hawkes": HawkesFlow,
    "queue_reactive": QueueReactiveFlow,
    "qr": QueueReactiveFlow,
}

_HAWKES_BASE: dict[str, Any] = {
    "beta": 5.0,
    "size_mu": 3.0,
    "size_sigma": 1.0,
    "depth_decay": 0.7,
    "max_depth": 10,
    "p_hidden": 0.02,
    "p_iceberg": 0.05,
}


def _hawkes(mu_scale: float, alpha_scale: float) -> dict[str, Any]:
    mu = np.array([1.2, 1.2, 0.35, 0.35, 0.9, 0.9]) * mu_scale
    alpha = (
        np.array(
            [
                # rows = excited dim; cols = exciting dim
                [0.8, 0.2, 0.1, 0.05, 0.1, 0.0],
                [0.2, 0.8, 0.05, 0.1, 0.0, 0.1],
                [0.2, 0.05, 0.6, 0.1, 0.05, 0.0],
                [0.05, 0.2, 0.1, 0.6, 0.0, 0.05],
                [0.2, 0.0, 0.05, 0.0, 0.5, 0.05],
                [0.0, 0.2, 0.0, 0.05, 0.05, 0.5],
            ]
        )
        * alpha_scale
    )
    return {**_HAWKES_BASE, "mu": mu, "alpha": alpha}


def _zi(lam_l: float, lam_m: float, mu_c: float) -> dict[str, Any]:
    return {
        "lambda_limit": lam_l,
        "lambda_market": lam_m,
        "mu_cancel": mu_c,
        "depth_decay": 0.7,
        "max_depth": 10,
        "p_inside": 0.3,
        "size_mu": 3.0,
        "size_sigma": 1.0,
        "p_hidden": 0.02,
        "p_iceberg": 0.05,
    }


def _qr(scale: float, market: float, qref: float) -> dict[str, Any]:
    return {
        "k": 3,
        "qref": qref,
        "lL": (2.5 * scale, 1.5 * scale, 1.0 * scale),
        "lC": (0.15, 0.15, 0.15),
        "lM": (market, 0.0, 0.0),
        "theta": 0.5,
        "size_mu": 3.0,
        "size_sigma": 0.5,
    }


# (flow_params, seed_book kwargs)
REGIMES: dict[str, dict[str, tuple[dict, dict]]] = {
    "calm": {
        "hawkes": (_hawkes(0.7, 0.6), {"mid_ticks": 10_000, "levels": 10, "qty_per_level": 60}),
        "zi": (_zi(0.7, 0.5, 0.04), {"mid_ticks": 10_000, "levels": 10, "qty_per_level": 40}),
        "queue_reactive": (
            _qr(0.8, 0.7, 40.0),
            {"mid_ticks": 10_000, "levels": 10, "qty_per_level": 40},
        ),
    },
    "normal": {
        "hawkes": (_hawkes(1.0, 1.0), {"mid_ticks": 10_000, "levels": 10, "qty_per_level": 40}),
        "zi": (_zi(1.2, 1.5, 0.05), {"mid_ticks": 10_000, "levels": 10, "qty_per_level": 20}),
        "queue_reactive": (
            _qr(1.0, 1.5, 30.0),
            {"mid_ticks": 10_000, "levels": 10, "qty_per_level": 20},
        ),
    },
    "volatile": {
        "hawkes": (_hawkes(2.0, 1.6), {"mid_ticks": 10_000, "levels": 10, "qty_per_level": 30}),
        "zi": (_zi(2.0, 3.5, 0.10), {"mid_ticks": 10_000, "levels": 10, "qty_per_level": 15}),
        "queue_reactive": (
            _qr(2.0, 3.0, 20.0),
            {"mid_ticks": 10_000, "levels": 10, "qty_per_level": 15},
        ),
    },
    "thin": {
        "hawkes": (_hawkes(0.6, 0.8), {"mid_ticks": 10_000, "levels": 10, "qty_per_level": 10}),
        "zi": (_zi(0.5, 0.8, 0.05), {"mid_ticks": 10_000, "levels": 10, "qty_per_level": 8}),
        "queue_reactive": (
            _qr(0.6, 0.9, 10.0),
            {"mid_ticks": 10_000, "levels": 10, "qty_per_level": 8},
        ),
    },
    # high-volume regime for the $10M/30-min execution benchmark:
    # limit posting must outrun market takers or the book starves (measured:
    # mu_l=1.2 leaves the bid side empty ~45% of the time and metaorders can't
    # fill). mu_l=4.0 keeps the contra side stocked ~97%, ~750k shares natural
    # one-sided volume per 1800 s -> a 100k-share metaorder is ~13%
    # participation.
    "exec": {
        "hawkes": (
            {
                **_hawkes(1.0, 1.0),
                "mu": np.array([4.0, 4.0, 3.0, 3.0, 0.9, 0.9]),
                "size_mu": 4.2,
            },
            {"mid_ticks": 10_000, "levels": 15, "qty_per_level": 120},
        ),
        "zi": (
            {**_zi(4.0, 3.0, 0.08), "size_mu": 4.2},
            {"mid_ticks": 10_000, "levels": 15, "qty_per_level": 120},
        ),
        "queue_reactive": (
            {**_qr(3.0, 4.0, 60.0), "size_mu": 4.2},
            {"mid_ticks": 10_000, "levels": 15, "qty_per_level": 120},
        ),
    },
}


def get_regime(name: str, flow: str = "hawkes") -> tuple[type, dict, dict]:
    """Return (flow_class, flow_params, seed_book_kwargs)."""
    flow_key = "queue_reactive" if flow == "qr" else flow
    if flow_key not in FLOWS:
        raise KeyError(f"unknown flow {flow!r}; have {sorted(FLOWS)}")
    if name not in REGIMES:
        raise KeyError(f"unknown regime {name!r}; have {sorted(REGIMES)}")
    params, seed_kwargs = REGIMES[name][flow_key]
    return FLOWS[flow_key], dict(params), dict(seed_kwargs)
