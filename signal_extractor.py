import re
from state import SessionState, TurnSignal

override_patterns = re.compile(
    r"\b(actually|instead|never\s?mind|change(d)? my mind|"
    r"ignore (that|my earlier|what i said))\b",
    re.IGNORECASE,
)
no_preference_patterns = re.compile(
    r"\b(no preference|don'?t (have|care|mind)|whatever you think|doesn'?t matter|"
    r"up to you|your judgement|whatever works|anything is fine)\b",
    re.IGNORECASE,
)
vague_patterns = re.compile(
    r"\b(not sure|maybe|perhaps|could be|just looking|unsure|"
    r"still exploring|browsing|just checking)\b",
    re.IGNORECASE,
)

material_patterns = re.compile(
    r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b",
    re.IGNORECASE,
)
colour_patterns = re.compile(
    r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b",
    re.IGNORECASE,
)
size_patterns = re.compile(
    r"\b(size|sizing|width|wide|narrow|gauge|diameter|length|height|inch)\b",
    re.IGNORECASE,
)
style_patterns = re.compile(
    r"\b(department|style|fit|sleeve|neck)\b",
    re.IGNORECASE,
)
use_case_patterns = re.compile(
    r"\b(hiking|running|gym|winter|outdoor|work)\b",
    re.IGNORECASE,
)

budget_patterns = re.compile(
    r"\b(budget|under \$?\d+|cheap(er)?|afford(able)?|inexpensive|price|"
    r"premium|luxury|expensive|high-end|high quality)\b",
    re.IGNORECASE,
)

BUCKET_PATTERNS = {
    "material": material_patterns,
    "colour": colour_patterns,
    "size": size_patterns,
    "style": style_patterns,
    "use_case": use_case_patterns,
    "budget": budget_patterns,
}

def find_disclosed_bucket(text: str) -> tuple[str|None, str|None]:
    # Look at the raw text and see if a real bucket is mentioned
    for bucket, pattern in BUCKET_PATTERNS.items():
        match = pattern.search(text)
        if match:
            return bucket, match.group(0)
    return None, None

def extract_signal(user_input: str, prior_state: SessionState) -> TurnSignal:
    text = user_input.strip()
    
    is_override = bool(override_patterns.search(text))
    disclosed_bucket, disclosed_value = find_disclosed_bucket(text)
    is_no_preference = bool(no_preference_patterns.search(text))
    confidence_delta = 0.0

    is_vague = bool(vague_patterns.search(text))

    # confidence delta
    if disclosed_bucket is not None:
        if prior_state.track == "browsing":
            confidence_delta = 0.5
        else:
            confidence_delta = 0.2 #already shopping

    elif is_vague:
        confidence_delta = -0.3

    elif is_no_preference:
        confidence_delta = 0.0

    # override pattern detection
    if is_override and disclosed_bucket is not None:
        confidence_delta = max(confidence_delta,0.5)

    # clamp confidence delta to [0.0, 1.0] range
    confidence_delta = max(0.0, min(1.0, confidence_delta)) if confidence_delta >= 0 else confidence_delta

    return TurnSignal(
        is_override = is_override,
        disclosed_bucket = disclosed_bucket,
        disclosed_value = disclosed_value,
        is_no_preference = is_no_preference,
        confidence_delta = confidence_delta
    )

def classify_track(state: SessionState) -> str:
    # Called after update_state()
    if state.buying_confidence > 0.8:
        return "buying"
    return "browsing"

# Never touches confirmed_constraints/exhausted_buckets directly.