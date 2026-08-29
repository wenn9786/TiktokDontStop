"""
Tests for signal_extractor.py

Run with:  pytest test_signal_extractor.py -v

These tests construct SessionState / TurnSignal objects directly, per the
project's testing plan — so they can run completely on their own, without
needing dialogue_manager.py, agent.py, or the evaluator.
"""

from state import SessionState
from signal_extractor import find_disclosed_bucket, extract_signal, classify_track


def make_state(
    track="browsing",
    buying_confidence=0.0,
    confirmed_constraints=None,
    exhausted_buckets=None,
    asked_buckets=None,
):
    """Small helper so each test only has to set the fields it cares about."""
    return SessionState(
        track=track,
        buying_confidence=buying_confidence,
        confirmed_constraints=confirmed_constraints or {},
        exhausted_buckets=exhausted_buckets or set(),
        asked_buckets=asked_buckets or set(),
    )


# ---------------------------------------------------------------------------
# find_disclosed_bucket
# ---------------------------------------------------------------------------

def test_finds_material_bucket():
    bucket, value = find_disclosed_bucket("It's 100% cotton and very soft.")
    assert bucket == "material"
    assert value.lower() == "cotton"


def test_finds_colour_bucket():
    bucket, value = find_disclosed_bucket("I'd like it in black please.")
    assert bucket == "colour"
    assert value.lower() == "black"


def test_finds_size_bucket():
    bucket, value = find_disclosed_bucket("What's the width of this band?")
    assert bucket == "size"
    assert value.lower() == "width"


def test_finds_style_bucket():
    # Deliberately avoids the word "in" here — see
    # test_bare_in_token_falsely_triggers_size_bucket below for why.
    bucket, value = find_disclosed_bucket("I need a slim style dress.")
    assert bucket == "style"
    assert value.lower() == "style"


def test_finds_use_case_bucket():
    bucket, value = find_disclosed_bucket("I need boots for hiking.")
    assert bucket == "use_case"
    assert value.lower() == "hiking"


def test_no_bucket_when_no_keyword_present():
    bucket, value = find_disclosed_bucket("I'm not sure what I want yet.")
    assert bucket is None
    assert value is None


def test_empty_string_returns_none():
    bucket, value = find_disclosed_bucket("")
    assert bucket is None
    assert value is None


def test_bucket_priority_when_multiple_keywords_present():
    # Text mentions both a material word ("cotton") and a colour word ("black").
    # BUCKET_PATTERNS is a dict, and Python dicts preserve insertion order, so
    # "material" is checked before "colour" and wins. This test documents that
    # ordering explicitly, since it's easy to break by accident later.
    bucket, value = find_disclosed_bucket("A black cotton scarf.")
    assert bucket == "material"
    assert value.lower() == "cotton"


# ---------------------------------------------------------------------------
# extract_signal — disclosure -> confidence_delta
# ---------------------------------------------------------------------------

def test_disclosure_while_browsing_gives_big_confidence_jump():
    state = make_state(track="browsing", buying_confidence=0.0)
    signal = extract_signal("I need it in leather.", state)
    assert signal.disclosed_bucket == "material"
    assert signal.disclosed_value.lower() == "leather"
    assert signal.confidence_delta == 0.5
    assert signal.is_override is False
    assert signal.is_no_preference is False


def test_disclosure_while_already_buying_gives_smaller_bump():
    state = make_state(track="buying", buying_confidence=0.85)
    signal = extract_signal("Make it blue.", state)
    assert signal.disclosed_bucket == "colour"
    assert signal.confidence_delta == 0.2


def test_vague_phrase_gives_negative_confidence_delta():
    state = make_state(track="browsing")
    signal = extract_signal("Not sure, just looking around.", state)
    assert signal.disclosed_bucket is None
    assert signal.confidence_delta == -0.3
    assert signal.is_override is False
    assert signal.is_no_preference is False


def test_negative_delta_is_not_clamped_to_zero():
    # The clamp logic only clamps non-negative deltas into [0, 1];
    # negative deltas pass through untouched. This test locks in that
    # (somewhat surprising) behavior so it doesn't change silently.
    state = make_state(track="browsing")
    signal = extract_signal("maybe, perhaps later", state)
    assert signal.confidence_delta < 0
    assert signal.confidence_delta == -0.3


def test_no_preference_phrase_sets_flag_with_zero_delta():
    state = make_state(track="browsing")
    signal = extract_signal("I don't have a preference, up to you.", state)
    assert signal.is_no_preference is True
    assert signal.disclosed_bucket is None
    assert signal.confidence_delta == 0.0


def test_completely_empty_input_is_a_no_op():
    state = make_state(track="browsing")
    signal = extract_signal("", state)
    assert signal.disclosed_bucket is None
    assert signal.is_override is False
    assert signal.is_no_preference is False
    assert signal.confidence_delta == 0.0


# ---------------------------------------------------------------------------
# extract_signal — override handling
# ---------------------------------------------------------------------------

def test_override_with_disclosed_value_boosts_confidence():
    state = make_state(track="buying", buying_confidence=0.9)
    signal = extract_signal("Actually, ignore that — I need nylon instead.", state)
    assert signal.is_override is True
    assert signal.disclosed_bucket == "material"
    assert signal.confidence_delta == 0.5


def test_override_phrase_alone_does_not_boost_confidence():
    # Known current limitation: the confidence bump for an override only
    # fires when a concrete attribute value is ALSO disclosed in the same
    # message. "Actually, never mind" with no attribute word present is
    # detected as an override, but confidence_delta stays at 0.0 because
    # disclosed_bucket is None. Documenting this so it's a deliberate,
    # visible decision rather than a silent gap.
    state = make_state(track="browsing")
    signal = extract_signal("Actually, never mind.", state)
    assert signal.is_override is True
    assert signal.disclosed_bucket is None
    assert signal.confidence_delta == 0.0


def test_override_and_no_preference_together():
    state = make_state(track="browsing")
    signal = extract_signal("Actually, I don't have a preference for that.", state)
    assert signal.is_override is True
    assert signal.is_no_preference is True
    assert signal.disclosed_bucket is None
    assert signal.confidence_delta == 0.0


# ---------------------------------------------------------------------------
# classify_track
# ---------------------------------------------------------------------------

def test_classify_track_high_confidence_is_buying():
    state = make_state(buying_confidence=0.81)
    assert classify_track(state) == "buying"


def test_classify_track_at_threshold_is_not_yet_buying():
    # Boundary check on the ">" comparison: exactly 0.8 should NOT count
    # as buying yet, since the code uses a strict "> 0.8".
    state = make_state(buying_confidence=0.8)
    assert classify_track(state) == "browsing"


def test_classify_track_low_confidence_is_browsing():
    state = make_state(buying_confidence=0.1)
    assert classify_track(state) == "browsing"


def test_classify_track_never_returns_boundary():
    # NOTE: this version of classify_track has no "boundary" branch at all
    # (unlike an earlier draft that checked exhausted_buckets). This test
    # documents that fact so the gap is visible rather than assumed away —
    # if "boundary" is supposed to be a reachable track, that's a real bug
    # to raise with whoever owns dialogue_manager.py / state.py.
    state = make_state(buying_confidence=0.0, exhausted_buckets={"color", "size"})
    assert classify_track(state) == "browsing"
