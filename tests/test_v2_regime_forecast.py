from beta_spy.v2_regime_forecast import REGIMES, forecast_regime


def _state(regime: str = "QUIET"):
    return {
        "ready": True,
        "regime": regime,
        "analog_count": 60,
        "effective_analogs": 38.0,
        "mean_proximity": 0.16,
        "conformal_scale": 1.05,
        "training_samples": 900,
        "p_big_15": 0.18,
        "p_big_30": 0.25,
        "p_up_15": 0.56,
        "p_up_30": 0.58,
        "p_reversal_15": 0.18,
        "p_reversal_30": 0.22,
        "p_persistent_30": 0.78,
        "p_acceleration": 0.30,
    }


def test_undefined_state_refuses_regime():
    result = forecast_regime({"ready": False})
    assert result.definable is False
    assert result.current_regime == "UNDEFINED"


def test_quiet_state_publishes_duration_and_normalized_successors():
    result = forecast_regime(_state())
    assert result.definable is True
    assert result.current_regime == "QUIET"
    assert result.persistence_15 > result.persistence_30
    assert 5.0 <= result.expected_duration_minutes <= 30.0
    assert set(result.successor_probabilities) == set(REGIMES)
    assert abs(sum(result.successor_probabilities.values()) - 1.0) < 1e-9
    assert result.successor_probabilities["QUIET"] > result.successor_probabilities["EXPANSION"]


def test_directional_up_assigns_nonzero_reversal_successor():
    state = _state("DIRECTIONAL_UP")
    state["p_reversal_30"] = 0.40
    state["p_up_30"] = 0.62
    result = forecast_regime(state)
    assert result.successor_probabilities["DIRECTIONAL_DOWN"] > 0.0
    assert result.persistence_30 <= result.persistence_15
