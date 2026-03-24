"""
Purpose:
- Handles job lifecycle records (queued/processing/completed/failed) in SQLite.

Libraries used:
- dataclasses/typing: typed job record object.
- db_cursor + state_machine: DB writes with validated state transitions.
"""

from dataclasses import dataclass
from typing import Optional

from app.services.db import db_cursor
from app.services.state_machine import validate_job_transition


@dataclass
class JobRecord:
    id: int
    file_id: Optional[int]
    job_type: str
    state: str
    attempt: int
    max_attempts: int
    error_message: Optional[str]
    created_at: str
    started_at: Optional[str]
    finished_at: Optional[str]
    updated_at: str


class JobRepository:
    def create(
        self,
        *,
        file_id: Optional[int],
        job_type: str,
        state: str = "queued",
        attempt: int = 0,
        max_attempts: int = 3,
        error_message: Optional[str] = None,
    ) -> JobRecord:
        with db_cursor() as cur:
            cur.execute(
                """
                INSERT INTO jobs (file_id, job_type, state, attempt, max_attempts, error_message)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (file_id, job_type, state, attempt, max_attempts, error_message),
            )
            job_id = cur.lastrowid
            cur.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
            row = cur.fetchone()

        return self._to_record(row)

    def get_by_id(self, job_id: int) -> Optional[JobRecord]:
        with db_cursor() as cur:
            cur.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
            row = cur.fetchone()

        return self._to_record(row) if row else None

    def transition(self, job_id: int, new_state: str, error_message: Optional[str] = None) -> JobRecord:
        job = self.get_by_id(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")

        validate_job_transition(job.state, new_state)

        started_at = "CURRENT_TIMESTAMP" if new_state == "processing" and not job.started_at else None
        finished_at = "CURRENT_TIMESTAMP" if new_state in {"completed", "failed", "dead_lettered"} else None

        with db_cursor() as cur:
            if started_at and finished_at:
                cur.execute(
                    """
                    UPDATE jobs
                    SET state = ?, error_message = ?, started_at = CURRENT_TIMESTAMP,
                        finished_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (new_state, error_message, job_id),
                )
            elif started_at:
                cur.execute(
                    """
                    UPDATE jobs
                    SET state = ?, error_message = ?, started_at = CURRENT_TIMESTAMP,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (new_state, error_message, job_id),
                )
            elif finished_at:
                cur.execute(
                    """
                    UPDATE jobs
                    SET state = ?, error_message = ?, finished_at = CURRENT_TIMESTAMP,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (new_state, error_message, job_id),
                )
            else:
                cur.execute(
                    """
                    UPDATE jobs
                    SET state = ?, error_message = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (new_state, error_message, job_id),
                )

            cur.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
            row = cur.fetchone()

        return self._to_record(row)

    def mark_failed(self, job_id: int, error_message: str) -> JobRecord:
        return self.transition(job_id, "failed", error_message=error_message)

    def increment_attempt(self, job_id: int) -> JobRecord:
        with db_cursor() as cur:
            cur.execute(
                """
                UPDATE jobs
                SET attempt = attempt + 1, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (job_id,),
            )
            cur.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
            row = cur.fetchone()

        return self._to_record(row)

    @staticmethod
    def _to_record(row) -> JobRecord:
        return JobRecord(
            id=row["id"],
            file_id=row["file_id"],
            job_type=row["job_type"],
            state=row["state"],
            attempt=row["attempt"],
            max_attempts=row["max_attempts"],
            error_message=row["error_message"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            updated_at=row["updated_at"],
        )
