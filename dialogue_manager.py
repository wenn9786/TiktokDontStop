from state import SessionState, TurnSignal

def update_state(prior: SessionState, signal: TurnSignal) -> SessionState:
    new_confirmed = dict(prior.confirmed_constraints)
    new_exhausted = set(prior.exhausted_buckets)
    new_asked = set(prior.asked_buckets)
    new_confidence = prior.buying_confidence
    new_mention_counts = dict(prior.bucket_mention_counts)

    if signal.is_override:
        # replace the old value for this bucket with the new one
        if signal.disclosed_bucket and signal.disclosed_value:
            new_confirmed[signal.disclosed_bucket] = signal.disclosed_value
            new_asked.add(signal.disclosed_bucket)
            new_mention_counts[signal.disclosed_bucket] = new_mention_counts.get(signal.disclosed_bucket, 0) + 1
        new_confidence += signal.confidence_delta

    elif signal.is_no_preference:
        if signal.disclosed_bucket:
            new_exhausted.add(signal.disclosed_bucket)
            new_asked.add(signal.disclosed_bucket)
        new_confidence += signal.confidence_delta

    elif signal.disclosed_bucket and signal.disclosed_value:
        new_confirmed[signal.disclosed_bucket] = signal.disclosed_value
        new_asked.add(signal.disclosed_bucket)
        new_confidence += signal.confidence_delta

    new_confidence = max(0.0, min(1.0, new_confidence))

    return SessionState(
        track=prior.track,
        buying_confidence=new_confidence,
        confirmed_constraints=new_confirmed,
        exhausted_buckets=new_exhausted,
        asked_buckets=new_asked,
        bucket_mention_counts=new_mention_counts,
        user_profile=prior.user_profile,
        shown_asins=prior.shown_asins,
        shown_counts=prior.shown_counts,
    )

def choose_next_attribute(state: SessionState, priority_order: list[str]) -> str | None:
    for bucket in priority_order:
        if bucket not in state.asked_buckets:
            return bucket
    return None

DEFAULT_PRIORITY = ["feature", "material", "color", "style", "size", "use_case", "budget"]

def build_query(state: SessionState) -> str:
    if not state.confirmed_constraints:
        return ""
    return " ".join(f"{value}" for value in state.confirmed_constraints.values())