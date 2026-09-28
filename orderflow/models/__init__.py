"""Neural prediction stack. Importing registers all predictors.

Torch-dependent modules are guarded: the book/sim/stats/execution core
deploys without torch (e.g. serverless), in which case the learned
predictors simply aren't registered.
"""

from orderflow.models import features, targets  # noqa: F401  (numpy-only)

try:
    from orderflow.models import (  # noqa: F401
        baselines,
        dataset,
        metrics,
        predictor,
        train,
        transformer,
    )
except ImportError:
    pass
