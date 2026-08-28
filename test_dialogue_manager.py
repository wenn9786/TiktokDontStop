# test_dialogue_manager.py
from state import SessionState, TurnSignal
from dialogue_manager import update_state, choose_next_attribute, build_query

def test_disclosed_value_gets_confirmed():
    prior = SessionState(track="browsing", buying_confidence=0.2)
    signal = TurnSignal(
        is_override=False,
        disclosed_bucket="material",
        disclosed_value="waterproof",
        is_no_preference=False,
        confidence_delta=0.3,
    )
    result = update_state(prior, signal)
    assert result.confirmed_constraints["material"] == "waterproof"
    assert "material" in result.asked_buckets
    assert result.buying_confidence == 0.5

def test_no_preference_goes_to_exhausted_not_confirmed():
    prior = SessionState(track="buying", buying_confidence=0.6)
    signal = TurnSignal(
        is_override=False,
        disclosed_bucket="brand",
        disclosed_value=None,
        is_no_preference=True,
        confidence_delta=0.0,
    )
    result = update_state(prior, signal)
    assert "brand" in result.exhausted_buckets
    assert "brand" not in result.confirmed_constraints

def test_choose_next_attribute_skips_asked():
    state = SessionState(
        track="buying",
        buying_confidence=0.5,
        asked_buckets={"feature"},
    )
    next_bucket = choose_next_attribute(state, ["feature", "material", "color"])
    assert next_bucket == "material"

def test_build_query_uses_only_confirmed():
    state = SessionState(
        track="buying",
        buying_confidence=0.7,
        confirmed_constraints={"material": "waterproof", "color": "black"},
    )
    query = build_query(state)
    assert "waterproof" in query
    assert "black" in query