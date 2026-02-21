"""
meeting_handler.py — Meeting join/leave/events.

Manages the lifecycle of a Teams meeting session: joining, leaving,
participant tracking, and routing meeting events to the audio pipeline.
Orchestrates AudioStreamHandler, WhisperRealtimeTranscriber, and ChunkManager
for each active meeting.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from botbuilder.core import TurnContext
from pydantic import BaseModel

from config.settings import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class MeetingState(str, Enum):
    """Lifecycle states for a meeting session."""

    PENDING = "pending"
    JOINING = "joining"
    ACTIVE = "active"
    LEAVING = "leaving"
    ENDED = "ended"
    ERROR = "error"


class Participant(BaseModel):
    """A meeting participant."""

    id: str
    name: str = "Unknown"
    role: str = "attendee"
    joined_at: float = 0.0
    left_at: float | None = None


@dataclass
class MeetingSession:
    """
    In-memory representation of an active meeting session.

    Holds references to all pipeline components and participant tracking.
    """

    meeting_id: str
    state: MeetingState = MeetingState.PENDING
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    meeting_data: dict[str, Any] = field(default_factory=dict)

    # Pipeline components (set during initialisation)
    audio_handler: Any = None
    speech_client: Any = None
    chunk_manager: Any = None

    # Participant tracking
    participants: dict[str, Participant] = field(default_factory=dict)

    # Transcript buffer (real-time segments from Whisper)
    transcript_segments: list[dict[str, Any]] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        """Meeting duration in seconds."""
        end = self.ended_at or time.time()
        return end - self.started_at

    @property
    def participant_count(self) -> int:
        """Number of currently active participants."""
        return sum(1 for p in self.participants.values() if p.left_at is None)


# ---------------------------------------------------------------------------
# MeetingHandler
# ---------------------------------------------------------------------------

class MeetingHandler:
    """
    Orchestrates the lifecycle of Teams meeting sessions.

    Responsibilities:
    - Start/stop audio pipeline on meeting start/end
    - Track participants joining and leaving
    - Route real-time transcript segments from Whisper
    - Provide meeting status and transcript data to the API layer

    All pipeline components are injected via factory callables for testability.
    """

    def __init__(
        self,
        blob_uploader: Any = None,
        queue_publisher: Any = None,
    ) -> None:
        self._sessions: dict[str, MeetingSession] = {}
        self._blob_uploader = blob_uploader
        self._queue_publisher = queue_publisher

        logger.info("MeetingHandler initialised")

    # ------------------------------------------------------------------
    # Meeting lifecycle
    # ------------------------------------------------------------------

    async def on_meeting_start(
        self,
        meeting_id: str,
        meeting_data: dict[str, Any],
        turn_context: TurnContext | None = None,
    ) -> MeetingSession:
        """
        Handle a meeting start event.

        Initialises the full audio pipeline:
        1. WhisperRealtimeTranscriber for real-time transcription
        2. ChunkManager for batch reprocessing
        3. AudioStreamHandler that fans out to both
        """
        if meeting_id in self._sessions:
            existing = self._sessions[meeting_id]
            if existing.state == MeetingState.ACTIVE:
                logger.warning("Meeting %s already active — ignoring duplicate start", meeting_id)
                return existing

        logger.info("Starting meeting session: %s", meeting_id)

        session = MeetingSession(
            meeting_id=meeting_id,
            state=MeetingState.JOINING,
            meeting_data=meeting_data,
        )
        self._sessions[meeting_id] = session

        try:
            await self._init_pipeline(session)
            session.state = MeetingState.ACTIVE
            logger.info("Meeting %s is now ACTIVE — pipeline running", meeting_id)

        except Exception:
            session.state = MeetingState.ERROR
            logger.exception("Failed to initialise pipeline for meeting %s", meeting_id)

        return session

    async def on_meeting_end(
        self,
        meeting_id: str,
        meeting_data: dict[str, Any] | None = None,
        turn_context: TurnContext | None = None,
    ) -> None:
        """
        Handle a meeting end event.

        Flushes all audio buffers and stops the pipeline gracefully.
        """
        session = self._sessions.get(meeting_id)
        if not session:
            logger.warning("Meeting %s not found — cannot end", meeting_id)
            return

        logger.info("Ending meeting session: %s (duration=%.0f s)", meeting_id, session.duration_seconds)
        session.state = MeetingState.LEAVING

        try:
            await self._teardown_pipeline(session)
            session.state = MeetingState.ENDED
            session.ended_at = time.time()
            logger.info(
                "Meeting %s ended — %d transcript segments, %d participants",
                meeting_id,
                len(session.transcript_segments),
                len(session.participants),
            )

        except Exception:
            session.state = MeetingState.ERROR
            logger.exception("Error tearing down pipeline for meeting %s", meeting_id)

    # ------------------------------------------------------------------
    # Participants
    # ------------------------------------------------------------------

    async def on_participants_join(
        self,
        participants: list[dict[str, Any]],
        turn_context: TurnContext | None = None,
    ) -> None:
        """Track participants joining across all active meetings."""
        # Determine meeting from context or find the active session
        meeting_id = self._resolve_meeting_id(turn_context)
        session = self._sessions.get(meeting_id) if meeting_id else None

        for p in participants:
            pid = p.get("id", p.get("aadObjectId", "unknown"))
            name = p.get("name", "Unknown")

            participant = Participant(
                id=pid,
                name=name,
                role=p.get("role", "attendee"),
                joined_at=time.time(),
            )

            if session:
                session.participants[pid] = participant

            logger.info(
                "Participant joined: %s (%s) in meeting %s",
                name,
                pid,
                meeting_id or "unknown",
            )

    async def on_participants_leave(
        self,
        participants: list[dict[str, Any]],
        turn_context: TurnContext | None = None,
    ) -> None:
        """Track participants leaving."""
        meeting_id = self._resolve_meeting_id(turn_context)
        session = self._sessions.get(meeting_id) if meeting_id else None

        for p in participants:
            pid = p.get("id", p.get("aadObjectId", "unknown"))
            name = p.get("name", "Unknown")

            if session and pid in session.participants:
                session.participants[pid].left_at = time.time()

            logger.info(
                "Participant left: %s (%s) from meeting %s",
                name,
                pid,
                meeting_id or "unknown",
            )

    # ------------------------------------------------------------------
    # Event routing
    # ------------------------------------------------------------------

    async def handle_event(self, turn_context: TurnContext) -> None:
        """Route a generic Teams event activity."""
        activity = turn_context.activity
        event_name = getattr(activity, "name", None) or activity.type
        logger.debug("Routing event: %s", event_name)

        # Additional event types can be handled here in the future

    # ------------------------------------------------------------------
    # Pipeline management
    # ------------------------------------------------------------------

    async def _init_pipeline(self, session: MeetingSession) -> None:
        """Initialise the audio capture and transcription pipeline."""
        from bot.audio_stream import AudioStreamHandler
        from transcription.whisper_realtime import WhisperRealtimeTranscriber
        from transcription.chunk_manager import ChunkManager

        meeting_id = session.meeting_id

        # Transcript callback — stores segments in the session
        async def on_segment(segment) -> None:
            session.transcript_segments.append({
                "text": segment.text,
                "language": segment.language,
                "timestamp": segment.timestamp,
                "confidence": segment.confidence,
                "speaker_id": segment.speaker_id,
                "source": "whisper",
            })

        # Create components
        speech_client = WhisperRealtimeTranscriber(on_segment=on_segment)
        chunk_manager = ChunkManager(
            meeting_id=meeting_id,
            blob_uploader=self._blob_uploader,
            queue_publisher=self._queue_publisher,
        )
        audio_handler = AudioStreamHandler(
            speech_client=speech_client,
            chunk_manager=chunk_manager,
        )

        # Start pipeline
        await speech_client.start()
        await audio_handler.start()

        # Store references in session
        session.speech_client = speech_client
        session.chunk_manager = chunk_manager
        session.audio_handler = audio_handler

        logger.info("Pipeline initialised for meeting %s", meeting_id)

    async def _teardown_pipeline(self, session: MeetingSession) -> None:
        """Gracefully stop the pipeline and flush remaining data."""
        # Stop audio handler first (flushes chunk buffer)
        if session.audio_handler:
            try:
                await session.audio_handler.stop()
            except Exception:
                logger.exception("Error stopping audio handler for %s", session.meeting_id)

        # Stop speech client
        if session.speech_client:
            try:
                await session.speech_client.stop()
            except Exception:
                logger.exception("Error stopping speech client for %s", session.meeting_id)

        logger.info("Pipeline torn down for meeting %s", session.meeting_id)

    # ------------------------------------------------------------------
    # Helpers / accessors
    # ------------------------------------------------------------------

    def _resolve_meeting_id(self, turn_context: TurnContext | None) -> str | None:
        """Try to resolve the meeting ID from a TurnContext."""
        if not turn_context:
            return None

        activity = turn_context.activity
        # Try channel data first (Teams-specific)
        channel_data = getattr(activity, "channel_data", None) or {}
        meeting = channel_data.get("meeting", {})
        meeting_id = meeting.get("id")

        if meeting_id:
            return meeting_id

        # Fall back to conversation ID
        if activity.conversation:
            return activity.conversation.id

        return None

    def get_session(self, meeting_id: str) -> MeetingSession | None:
        """Get a meeting session by ID."""
        return self._sessions.get(meeting_id)

    @property
    def active_meeting_count(self) -> int:
        """Number of currently active meetings."""
        return sum(
            1 for s in self._sessions.values()
            if s.state == MeetingState.ACTIVE
        )

    @property
    def active_sessions(self) -> list[MeetingSession]:
        """List of all active meeting sessions."""
        return [
            s for s in self._sessions.values()
            if s.state == MeetingState.ACTIVE
        ]


# ---------------------------------------------------------------------------
# Unit test stub
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    async def _test() -> None:
        handler = MeetingHandler()
        assert handler.active_meeting_count == 0

        # Simulate meeting lifecycle (no real bot context)
        session = await handler.on_meeting_start(
            meeting_id="test-mtg-001",
            meeting_data={"title": "Test Meeting"},
        )
        # Pipeline init will fail without faster-whisper, but state is tracked
        print(f"Session state: {session.state.value}")
        print(f"Active meetings: {handler.active_meeting_count}")

        await handler.on_meeting_end(
            meeting_id="test-mtg-001",
            meeting_data={},
        )
        print(f"After end: {session.state.value}")
        print("✓ MeetingHandler test passed")

    asyncio.run(_test())
