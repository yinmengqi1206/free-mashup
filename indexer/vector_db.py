from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np


@dataclass
class ClipRow:
    clip_id: int
    video_path: str
    start: float
    end: float
    duration: float
    motion: float
    brightness: float
    color_mean: Tuple[float, float, float]
    color_std: Tuple[float, float, float]
    action: str
    action_score: float
    emotion: str
    emotion_score: float
    embedding: np.ndarray
    motion_curve: np.ndarray
    brightness_curve: np.ndarray


def _connect(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _ensure_columns(conn: sqlite3.Connection) -> None:
    cur = conn.execute("PRAGMA table_info(clips)")
    existing = {row[1] for row in cur.fetchall()}

    def add_column(name: str, ddl: str) -> None:
        if name not in existing:
            conn.execute(f"ALTER TABLE clips ADD COLUMN {ddl}")

    add_column("action", "action TEXT DEFAULT 'unknown'")
    add_column("action_score", "action_score REAL DEFAULT 0.0")
    add_column("emotion", "emotion TEXT DEFAULT 'unknown'")
    add_column("emotion_score", "emotion_score REAL DEFAULT 0.0")
    add_column("color_mean", "color_mean BLOB")
    add_column("color_std", "color_std BLOB")
    add_column("motion_curve", "motion_curve BLOB")
    add_column("brightness_curve", "brightness_curve BLOB")


def init_db(db_path: str) -> None:
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS videos (
                video_path TEXT PRIMARY KEY,
                mtime REAL,
                size INTEGER,
                processed_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS clips (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_path TEXT,
                start REAL,
                end REAL,
                duration REAL,
                motion REAL,
                brightness REAL,
                embedding BLOB,
                action TEXT DEFAULT 'unknown',
                action_score REAL DEFAULT 0.0,
                emotion TEXT DEFAULT 'unknown',
                emotion_score REAL DEFAULT 0.0,
                color_mean BLOB,
                color_std BLOB,
                motion_curve BLOB,
                brightness_curve BLOB
            )
            """
        )
        _ensure_columns(conn)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_clips_video ON clips(video_path)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_clips_duration ON clips(duration)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_clips_motion ON clips(motion)")
        conn.commit()
    finally:
        conn.close()


def video_needs_processing(db_path: str, video_path: str, mtime: float, size: int) -> bool:
    conn = _connect(db_path)
    try:
        cur = conn.execute(
            "SELECT mtime, size FROM videos WHERE video_path = ?",
            (video_path,),
        )
        row = cur.fetchone()
        if row is None:
            return True
        return float(row[0]) != float(mtime) or int(row[1]) != int(size)
    finally:
        conn.close()


def upsert_video(db_path: str, video_path: str, mtime: float, size: int, processed_at: str) -> None:
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO videos (video_path, mtime, size, processed_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(video_path) DO UPDATE SET mtime=excluded.mtime, size=excluded.size, processed_at=excluded.processed_at
            """,
            (video_path, mtime, size, processed_at),
        )
        conn.commit()
    finally:
        conn.close()


def delete_clips_for_video(db_path: str, video_path: str) -> None:
    conn = _connect(db_path)
    try:
        conn.execute("DELETE FROM clips WHERE video_path = ?", (video_path,))
        conn.commit()
    finally:
        conn.close()


def add_clips(
    db_path: str,
    rows: Iterable[
        Tuple[
            str,
            float,
            float,
            float,
            float,
            float,
            str,
            float,
            str,
            float,
            Tuple[float, float, float],
            Tuple[float, float, float],
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ]
    ],
) -> None:
    conn = _connect(db_path)
    try:
        data = []
        for (
            video_path,
            start,
            end,
            duration,
            motion,
            brightness,
            action,
            action_score,
            emotion,
            emotion_score,
            color_mean,
            color_std,
            embedding,
            motion_curve,
            brightness_curve,
        ) in rows:
            data.append(
                (
                    video_path,
                    float(start),
                    float(end),
                    float(duration),
                    float(motion),
                    float(brightness),
                    embedding.astype(np.float32).tobytes(),
                    action,
                    float(action_score),
                    emotion,
                    float(emotion_score),
                    np.asarray(color_mean, dtype=np.float32).tobytes(),
                    np.asarray(color_std, dtype=np.float32).tobytes(),
                    np.asarray(motion_curve, dtype=np.float32).tobytes(),
                    np.asarray(brightness_curve, dtype=np.float32).tobytes(),
                )
            )
        conn.executemany(
            """
            INSERT INTO clips (
                video_path, start, end, duration, motion, brightness, embedding,
                action, action_score, emotion, emotion_score,
                color_mean, color_std, motion_curve, brightness_curve
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            data,
        )
        conn.commit()
    finally:
        conn.close()


def _load_clips(db_path: str) -> List[ClipRow]:
    conn = _connect(db_path)
    try:
        cur = conn.execute(
            """
            SELECT id, video_path, start, end, duration, motion, brightness,
                   action, action_score, emotion, emotion_score,
                   color_mean, color_std, embedding, motion_curve, brightness_curve
            FROM clips
            """
        )
        rows = []
        for row in cur.fetchall():
            color_mean = np.frombuffer(row[11], dtype=np.float32) if row[11] else np.zeros(3, dtype=np.float32)
            color_std = np.frombuffer(row[12], dtype=np.float32) if row[12] else np.zeros(3, dtype=np.float32)
            emb = np.frombuffer(row[13], dtype=np.float32)
            motion_curve = np.frombuffer(row[14], dtype=np.float32) if row[14] else np.zeros(1, dtype=np.float32)
            brightness_curve = np.frombuffer(row[15], dtype=np.float32) if row[15] else np.zeros(1, dtype=np.float32)
            rows.append(
                ClipRow(
                    clip_id=int(row[0]),
                    video_path=row[1],
                    start=float(row[2]),
                    end=float(row[3]),
                    duration=float(row[4]),
                    motion=float(row[5]),
                    brightness=float(row[6]),
                    action=row[7] or "unknown",
                    action_score=float(row[8] or 0.0),
                    emotion=row[9] or "unknown",
                    emotion_score=float(row[10] or 0.0),
                    color_mean=(float(color_mean[0]), float(color_mean[1]), float(color_mean[2])),
                    color_std=(float(color_std[0]), float(color_std[1]), float(color_std[2])),
                    embedding=emb,
                    motion_curve=motion_curve,
                    brightness_curve=brightness_curve,
                )
            )
        return rows
    finally:
        conn.close()


def list_clips(db_path: str) -> List[ClipRow]:
    return _load_clips(db_path)
