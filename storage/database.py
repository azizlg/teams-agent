"""
database.py — PostgreSQL with SQLAlchemy.

Defines the async SQLAlchemy engine, session factory, and ORM models
for meetings, transcript segments, audio chunks, and speaker profiles.
Uses SQLAlchemy 2.0 async patterns with asyncpg.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import AsyncGenerator
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from config.settings import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all ORM models."""
    pass


# ---------------------------------------------------------------------------
# ORM Models
# ---------------------------------------------------------------------------

class Meeting(Base):
    """A recorded Teams meeting session."""

    __tablename__ = "meetings"

    id: Mapped[str] = mapped_column(
        String(64), primary_key=True, default=lambda: uuid4().hex[:12],
    )
    title: Mapped[str | None] = mapped_column(String(500), nullable=True)
    meeting_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), default="pending", index=True,
    )

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    ended_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )

    participant_count: Mapped[int] = mapped_column(Integer, default=0)
    languages_detected: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    metadata_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(),
    )

    # Relationships
    transcript_segments: Mapped[list["TranscriptSegment"]] = relationship(
        back_populates="meeting", cascade="all, delete-orphan",
    )
    audio_chunks: Mapped[list["AudioChunk"]] = relationship(
        back_populates="meeting", cascade="all, delete-orphan",
    )
    speakers: Mapped[list["Speaker"]] = relationship(
        back_populates="meeting", cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Meeting id={self.id} status={self.status}>"


class TranscriptSegment(Base):
    """A single transcript segment (from Azure Speech or Whisper)."""

    __tablename__ = "transcript_segments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    meeting_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("meetings.id", ondelete="CASCADE"), index=True,
    )
    chunk_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    sequence_number: Mapped[int] = mapped_column(Integer, default=0)

    text: Mapped[str] = mapped_column(Text, default="")
    language: Mapped[str] = mapped_column(String(10), default="unknown")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)

    start_time: Mapped[float] = mapped_column(Float, default=0.0)
    end_time: Mapped[float] = mapped_column(Float, default=0.0)
    timestamp: Mapped[float] = mapped_column(Float, default=0.0)

    speaker_id: Mapped[str | None] = mapped_column(String(50), nullable=True)
    source: Mapped[str] = mapped_column(
        String(20), default="azure",  # "azure" or "whisper"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
    )

    # Relationships
    meeting: Mapped["Meeting"] = relationship(back_populates="transcript_segments")

    # Indexes for common queries
    __table_args__ = (
        Index("ix_transcript_meeting_seq", "meeting_id", "sequence_number"),
        Index("ix_transcript_meeting_source", "meeting_id", "source"),
    )

    def __repr__(self) -> str:
        return f"<TranscriptSegment id={self.id} lang={self.language} source={self.source}>"


class AudioChunk(Base):
    """Metadata for an audio chunk stored in Blob Storage."""

    __tablename__ = "audio_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chunk_id: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    meeting_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("meetings.id", ondelete="CASCADE"), index=True,
    )
    sequence_number: Mapped[int] = mapped_column(Integer, default=0)

    blob_url: Mapped[str] = mapped_column(Text, default="")
    duration_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    sample_rate: Mapped[int] = mapped_column(Integer, default=16000)
    channels: Mapped[int] = mapped_column(Integer, default=1)

    status: Mapped[str] = mapped_column(
        String(20), default="pending",  # pending | processing | completed | failed
        index=True,
    )
    whisper_processed: Mapped[bool] = mapped_column(Boolean, default=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
    )
    processed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )

    # Relationships
    meeting: Mapped["Meeting"] = relationship(back_populates="audio_chunks")

    __table_args__ = (
        Index("ix_chunk_meeting_seq", "meeting_id", "sequence_number"),
    )

    def __repr__(self) -> str:
        return f"<AudioChunk chunk_id={self.chunk_id} status={self.status}>"


class Speaker(Base):
    """Speaker profile for a meeting (from diarization)."""

    __tablename__ = "speakers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    meeting_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("meetings.id", ondelete="CASCADE"), index=True,
    )

    speaker_label: Mapped[str] = mapped_column(String(50))  # pyannote label
    participant_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    participant_name: Mapped[str | None] = mapped_column(String(200), nullable=True)

    total_speaking_time: Mapped[float] = mapped_column(Float, default=0.0)
    segment_count: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
    )

    # Relationships
    meeting: Mapped["Meeting"] = relationship(back_populates="speakers")

    __table_args__ = (
        Index("ix_speaker_meeting_label", "meeting_id", "speaker_label", unique=True),
    )

    def __repr__(self) -> str:
        return f"<Speaker label={self.speaker_label} name={self.participant_name}>"


# ---------------------------------------------------------------------------
# Database engine & session factory
# ---------------------------------------------------------------------------

class Database:
    """
    Async database manager.

    Creates the SQLAlchemy async engine, session factory,
    and provides convenience methods for common operations.

    Usage:
        db = Database()
        await db.connect()
        async with db.session() as session:
            ...
        await db.close()
    """

    def __init__(
        self,
        *,
        url: str | None = None,
        pool_size: int | None = None,
        max_overflow: int | None = None,
    ) -> None:
        self._url = url or settings.database.url
        self._pool_size = pool_size or settings.database.pool_size
        self._max_overflow = max_overflow or settings.database.pool_max_overflow

        self._engine = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None

        logger.info("Database initialised — url=%s", self._url.split("@")[-1])

    async def connect(self) -> None:
        """Create the async engine and session factory."""
        self._engine = create_async_engine(
            self._url,
            pool_size=self._pool_size,
            max_overflow=self._max_overflow,
            echo=False,
        )
        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        logger.info("Database engine created")

    async def create_tables(self) -> None:
        """Create all tables (for development/testing)."""
        if not self._engine:
            raise RuntimeError("Not connected — call connect() first")

        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created")

    async def drop_tables(self) -> None:
        """Drop all tables (for testing only)."""
        if not self._engine:
            raise RuntimeError("Not connected — call connect() first")

        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        logger.info("Database tables dropped")

    async def close(self) -> None:
        """Dispose the engine and close all connections."""
        if self._engine:
            await self._engine.dispose()
            logger.info("Database engine disposed")

    def session(self) -> AsyncSession:
        """Get a new async session (use as async context manager)."""
        if not self._session_factory:
            raise RuntimeError("Not connected — call connect() first")
        return self._session_factory()

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    async def save_meeting(self, meeting: Meeting) -> Meeting:
        """Save or update a meeting record."""
        async with self.session() as session:
            session.add(meeting)
            await session.commit()
            await session.refresh(meeting)
            logger.debug("Meeting saved: %s", meeting.id)
            return meeting

    async def get_meeting(self, meeting_id: str) -> Meeting | None:
        """Get a meeting by ID."""
        async with self.session() as session:
            return await session.get(Meeting, meeting_id)

    async def save_transcript_segment(
        self,
        segment: TranscriptSegment,
    ) -> TranscriptSegment:
        """Save a transcript segment."""
        async with self.session() as session:
            session.add(segment)
            await session.commit()
            await session.refresh(segment)
            return segment

    async def save_transcript_batch(
        self,
        segments: list[TranscriptSegment],
    ) -> None:
        """Save multiple transcript segments in one transaction."""
        async with self.session() as session:
            session.add_all(segments)
            await session.commit()
            logger.debug("Saved %d transcript segments", len(segments))

    async def get_meeting_transcript(
        self,
        meeting_id: str,
        *,
        source: str | None = None,
    ) -> list[TranscriptSegment]:
        """
        Get all transcript segments for a meeting, ordered by sequence.

        Args:
            meeting_id: The meeting ID.
            source: Optional filter by source ("azure" or "whisper").
        """
        from sqlalchemy import select

        async with self.session() as session:
            stmt = (
                select(TranscriptSegment)
                .where(TranscriptSegment.meeting_id == meeting_id)
                .order_by(TranscriptSegment.sequence_number, TranscriptSegment.start_time)
            )
            if source:
                stmt = stmt.where(TranscriptSegment.source == source)

            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def save_audio_chunk(self, chunk: AudioChunk) -> AudioChunk:
        """Save audio chunk metadata."""
        async with self.session() as session:
            session.add(chunk)
            await session.commit()
            await session.refresh(chunk)
            return chunk

    async def update_chunk_status(
        self,
        chunk_id: str,
        status: str,
        *,
        error_message: str | None = None,
    ) -> None:
        """Update the processing status of an audio chunk."""
        from sqlalchemy import update

        async with self.session() as session:
            stmt = (
                update(AudioChunk)
                .where(AudioChunk.chunk_id == chunk_id)
                .values(
                    status=status,
                    whisper_processed=(status == "completed"),
                    error_message=error_message,
                    processed_at=func.now() if status in ("completed", "failed") else None,
                )
            )
            await session.execute(stmt)
            await session.commit()
            logger.debug("Chunk %s status → %s", chunk_id, status)

    async def save_speaker(self, speaker: Speaker) -> Speaker:
        """Save or update a speaker profile."""
        async with self.session() as session:
            session.add(speaker)
            await session.commit()
            await session.refresh(speaker)
            return speaker

    @property
    def is_connected(self) -> bool:
        return self._engine is not None


# ---------------------------------------------------------------------------
# Unit test stub
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    async def _test() -> None:
        db = Database()
        assert not db.is_connected

        # Verify models can be instantiated
        meeting = Meeting(id="test-1", title="Test Meeting", status="active")
        print(f"Meeting: {meeting}")

        segment = TranscriptSegment(
            meeting_id="test-1",
            text="Hello world",
            language="en-US",
            confidence=0.95,
            source="azure",
        )
        print(f"Segment: {segment}")

        chunk = AudioChunk(
            chunk_id="test-1_0001_abc12345",
            meeting_id="test-1",
            sequence_number=1,
            blob_url="https://blob.example.com/test.wav",
            duration_seconds=300.0,
        )
        print(f"Chunk: {chunk}")

        speaker = Speaker(
            meeting_id="test-1",
            speaker_label="SPEAKER_00",
            participant_name="Alice",
            total_speaking_time=120.5,
        )
        print(f"Speaker: {speaker}")

        print("✓ Database models test passed")
        print("  (Full test requires PostgreSQL connection)")

    asyncio.run(_test())
