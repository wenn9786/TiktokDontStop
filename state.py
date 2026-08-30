from dataclasses import dataclass, field

@dataclass
class SessionState:
    track: str = "browsing"
    buying_confidence: float = 0.0
    confirmed_constraints: dict[str, str] = field(default_factory=dict)
    exhausted_buckets: set[str] = field(default_factory=set)
    asked_buckets: set[str] = field(default_factory=set)
    bucket_mention_counts: dict[str, int] = field(default_factory=dict)
    user_profile: dict = field(default_factory=dict)
    shown_asins: dict[str, int] = field(default_factory=dict)
    shown_counts: dict[str, int] = field(default_factory=dict)



@dataclass
class TurnSignal:
    is_override: bool
    disclosed_bucket: str | None
    disclosed_value: str | None
    is_no_preference: bool
    confidence_delta: float