"""
Purpose:
- Handles CRUD for artifact metadata records in SQLite.

Libraries used:
- dataclasses/typing: typed record object.
- app.services.db.db_cursor: shared DB cursor/transaction helper.
"""

from dataclasses import dataclass
from typing import Optional

from app.services.db import db_cursor


@dataclass
class ArtifactRecord:
    id: int
    file_id: Optional[int]
    artifact_type: str
    storage_backend: str
    bucket: Optional[str]
    object_key: str
    etag: Optional[str]
    content_type: Optional[str]
    size: Optional[int]
    version_id: Optional[str]
    created_at: str
    updated_at: str


class ArtifactRepository:
    def create(
        self,
        *,
        file_id: Optional[int],
        artifact_type: str,
        storage_backend: str,
        object_key: str,
        bucket: Optional[str] = None,
        etag: Optional[str] = None,
        content_type: Optional[str] = None,
        size: Optional[int] = None,
        version_id: Optional[str] = None,
    ) -> ArtifactRecord:
        with db_cursor() as cur:
            cur.execute(
                """
                INSERT INTO artifacts (
                    file_id, artifact_type, storage_backend, bucket, object_key,
                    etag, content_type, size, version_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (file_id, artifact_type, storage_backend, bucket, object_key, etag, content_type, size, version_id),
            )
            artifact_id = cur.lastrowid
            cur.execute("SELECT * FROM artifacts WHERE id = ?", (artifact_id,))
            row = cur.fetchone()

        return self._to_record(row)

    @staticmethod
    def _to_record(row) -> ArtifactRecord:
        return ArtifactRecord(
            id=row["id"],
            file_id=row["file_id"],
            artifact_type=row["artifact_type"],
            storage_backend=row["storage_backend"],
            bucket=row["bucket"],
            object_key=row["object_key"],
            etag=row["etag"],
            content_type=row["content_type"],
            size=row["size"],
            version_id=row["version_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
