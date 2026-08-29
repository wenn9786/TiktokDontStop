from dataclasses import dataclass

@dataclass
class SessionState:
    track: str
    buying_confidence: float
    confirmed_constraints: dict[str, str]
    exhausted_buckets: set[str]
    asked_buckets: set[str]

@dataclass
class TurnSignal:
    is_override: bool
    disclosed_bucket: str | None
    disclosed_value: str | None
    is_no_preference: bool
    confidence_delta: float