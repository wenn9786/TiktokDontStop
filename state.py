from dataclasses import dataclass, field

@dataclass
class SessionState:
    track: str = "browsing"
    buying_confidence: float = 0.0
    confirmed_constraints: dict[str, str] = field(default_factory=dict)
    exhausted_buckets: set[str] = field(default_factory=set)
    asked_buckets: set[str] = field(default_factory=set)

@dataclass
class TurnSignal:
    is_override: bool
    disclosed_bucket: str | None
    disclosed_value: str | None
    is_no_preference: bool
    confidence_delta: float