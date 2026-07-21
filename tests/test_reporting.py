from __future__ import annotations

import pandas as pd
import pytest

from compass.reporting import summarize_hierarchical_run


def test_summarize_hierarchical_run_checks_context_ranking_and_cv_gain():
    metadata = {
        "best_lambda": 10.0,
        "cv_scores": {"1000000.0": 5.0, "10.0": 4.0, "1.0": 4.5},
    }
    contributions = pd.DataFrame(
        {
            "context": ["intercept", "microglia", "neuron"],
            "total_h2": [0.2, 0.6, 0.2],
            "fraction": [0.2, 0.6, 0.2],
        }
    )
    result = summarize_hierarchical_run(metadata, contributions, "microglia")
    assert result["best_lambda"] == 10.0
    assert result["cv_mse_improvement"] == 1.0
    assert result["leading_context"] == "microglia"
    assert result["leading_fraction_biological"] == pytest.approx(0.75)
    assert result["expected_context_is_leading"]
