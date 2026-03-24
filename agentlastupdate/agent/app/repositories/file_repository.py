"""
Purpose:
- Handles CRUD for source file metadata records in SQLite.

Libraries used:
- dataclasses/typing: typed file record object.
- app.services.db.db_cursor: shared DB cursor/transaction helper.
"""

from dataclasses import dataclass
from typing import Optional

from app.services.db import db_cursor


@dataclass
class FileRecord:
    id: int
    original_name: str
    checksum: str
    size: int
    mime: str
    classification: str
    status: str
    error_message: Optional[str]
    created_at: str
    updated_at: str


class FileRepository:
    def get_by_checksum(self, checksum: str) -> Optional[FileRecord]:
        with db_cursor() as cur:
            cur.execute("SELECT * FROM files WHERE checksum = ?", (checksum,))
            row = cur.fetchone()

        return self._to_record(row) if row else None

    def create(
        self,
        *,
        original_name: str,
        checksum: str,
        size: int,
        mime: str,
        classification: str,
        status: str,
        error_message: Optional[str] = None,
    ) -> FileRecord:
        with db_cursor() as cur:
            cur.execute(
                """
                INSERT INTO files (
                    original_name, checksum, size, mime, classification, status, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (original_name, checksum, size, mime, classification, status, error_message),
            )
            file_id = cur.lastrowid
            cur.execute("SELECT * FROM files WHERE id = ?", (file_id,))
            row = cur.fetchone()

        return self._to_record(row)

    def update_status(self, file_id: int, status: str, error_message: Optional[str] = None) -> None:
        with db_cursor() as cur:
            cur.execute(
                """
                UPDATE files
                SET status = ?, error_message = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (status, error_message, file_id),
            )

    @staticmethod
    def _to_record(row) -> FileRecord:
        return FileRecord(
            id=row["id"],
            original_name=row["original_name"],
            checksum=row["checksum"],
            size=row["size"],
            mime=row["mime"] or "",
            classification=row["classification"],
            status=row["status"],
            error_message=row["error_message"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
