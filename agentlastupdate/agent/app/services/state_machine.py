"""
Purpose:
- Defines and validates allowed job state transitions.

Libraries used:
- Built-in types only; keeps workflow states consistent.
"""

ALLOWED_JOB_TRANSITIONS: dict[str, set[str]] = {
    "queued": {"processing", "failed", "dead_lettered"},
    "processing": {"completed", "failed", "dead_lettered"},
    "failed": {"queued", "dead_lettered"},
    "completed": set(),
    "dead_lettered": set(),
}


def validate_job_transition(current_state: str, new_state: str) -> None:
    allowed = ALLOWED_JOB_TRANSITIONS.get(current_state, set())
    if new_state not in allowed:
        raise ValueError(
            f"Invalid job state transition: {current_state} -> {new_state}. "
            f"Allowed: {sorted(allowed)}"
        )
