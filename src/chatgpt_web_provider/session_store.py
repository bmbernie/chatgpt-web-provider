from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path


STORE_VERSION = 1


@dataclass(frozen=True, slots=True)
class PersistedSession:
    session_id: str
    model: str
    level: str
    conversation_policy: str
    conversation_url: str


class SessionStore:
    def __init__(
        self,
        path: str | Path,
    ):
        self.path = Path(path).expanduser()

    def load(
        self,
    ) -> dict[str, PersistedSession]:
        if not self.path.exists():
            return {}

        with self.path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            payload = json.load(handle)

        if not isinstance(payload, dict):
            raise ValueError(
                "invalid session-state root"
            )

        if payload.get("version") != STORE_VERSION:
            raise ValueError(
                "unsupported session-state version"
            )

        raw_sessions = payload.get(
            "sessions"
        )

        if not isinstance(
            raw_sessions,
            dict,
        ):
            raise ValueError(
                "invalid session-state sessions"
            )

        result = {}

        for session_id, raw in (
            raw_sessions.items()
        ):
            if not isinstance(
                session_id,
                str,
            ):
                raise ValueError(
                    "invalid persisted session id"
                )

            if not isinstance(raw, dict):
                raise ValueError(
                    "invalid persisted session"
                )

            record = PersistedSession(
                session_id=str(
                    raw["session_id"]
                ),
                model=str(raw["model"]),
                level=str(raw["level"]),
                conversation_policy=str(
                    raw[
                        "conversation_policy"
                    ]
                ),
                conversation_url=str(
                    raw["conversation_url"]
                ),
            )

            if (
                record.session_id
                != session_id
            ):
                raise ValueError(
                    "persisted session id mismatch"
                )

            result[session_id] = record

        return result

    def save(
        self,
        sessions: dict[
            str,
            PersistedSession,
        ],
    ) -> None:
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
            mode=0o700,
        )

        payload = {
            "version": STORE_VERSION,
            "sessions": {
                session_id: asdict(
                    sessions[session_id]
                )
                for session_id
                in sorted(sessions)
            },
        }

        temp_path = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=(
                    self.path.name + "."
                ),
                suffix=".tmp",
                delete=False,
            ) as handle:
                temp_path = Path(
                    handle.name
                )

                os.chmod(
                    temp_path,
                    0o600,
                )

                json.dump(
                    payload,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(
                    handle.fileno()
                )

            os.replace(
                temp_path,
                self.path,
            )
            temp_path = None

            os.chmod(
                self.path,
                0o600,
            )

        finally:
            if (
                temp_path is not None
                and temp_path.exists()
            ):
                temp_path.unlink()

    def upsert(
        self,
        record: PersistedSession,
    ) -> None:
        sessions = self.load()
        sessions[
            record.session_id
        ] = record
        self.save(sessions)

    def delete(
        self,
        session_id: str,
    ) -> None:
        sessions = self.load()

        if (
            sessions.pop(
                session_id,
                None,
            )
            is None
        ):
            return

        self.save(sessions)
