"""
Purpose:
- Handles final report run tracking records in SQLite.

Libraries used:
- dataclasses/typing: typed run record object.
- app.services.db.db_cursor: shared DB cursor/transaction helper.
"""

from dataclasses import dataclass
from typing import Optional

from app.services.db import db_cursor


@dataclass
class FinalReportRunRecord:
    id: int
    trigger_type: str
    state: str
    input_count: int
    output_artifact_id: Optional[int]
    error_message: Optional[str]
    created_at: str
    started_at: Optional[str]
    finished_at: Optional[str]
    updated_at: str


class RunRepository:
    def create(
        self,
        *,
        trigger_type: str,
        state: str,
        input_count: int,
        output_artifact_id: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> FinalReportRunRecord:
        with db_cursor() as cur:
            cur.execute(
                """
                INSERT INTO final_report_runs (
                    trigger_type, state, input_count, output_artifact_id, error_message
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (trigger_type, state, input_count, output_artifact_id, error_message),
            )
            run_id = cur.lastrowid
            cur.execute("SELECT * FROM final_report_runs WHERE id = ?", (run_id,))
            row = cur.fetchone()

        return self._to_record(row)

    def mark_processing(self, run_id: int) -> FinalReportRunRecord:
        with db_cursor() as cur:
            cur.execute(
                """
                UPDATE final_report_runs
                SET state = 'processing', started_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (run_id,),
            )
            cur.execute("SELECT * FROM final_report_runs WHERE id = ?", (run_id,))
            row = cur.fetchone()

        return self._to_record(row)

    def mark_completed(self, run_id: int, output_artifact_id: Optional[int]) -> FinalReportRunRecord:
        with db_cursor() as cur:
            cur.execute(
                """
                UPDATE final_report_runs
                SET state = 'completed', output_artifact_id = ?, finished_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (output_artifact_id, run_id),
            )
            cur.execute("SELECT * FROM final_report_runs WHERE id = ?", (run_id,))
            row = cur.fetchone()

        return self._to_record(row)

    def mark_failed(self, run_id: int, error_message: str) -> FinalReportRunRecord:
        with db_cursor() as cur:
            cur.execute(
                """
                UPDATE final_report_runs
                SET state = 'failed', error_message = ?, finished_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (error_message, run_id),
            )
            cur.execute("SELECT * FROM final_report_runs WHERE id = ?", (run_id,))
            row = cur.fetchone()

        return self._to_record(row)

    @staticmethod
    def _to_record(row) -> FinalReportRunRecord:
        return FinalReportRunRecord(
            id=row["id"],
            trigger_type=row["trigger_type"],
            state=row["state"],
            input_count=row["input_count"],
            output_artifact_id=row["output_artifact_id"],
            error_message=row["error_message"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            updated_at=row["updated_at"],
        )
