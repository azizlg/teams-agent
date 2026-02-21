"""
audio_stream.py — Real-time audio stream capture from Teams RTP.

AudioStreamHandler receives raw PCM audio frames from the Teams meeting,
fans them out to the Azure Speech real-time transcription client and
the ChunkManager buffer for Whisper reprocessing. All I/O is non-blocking.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Protocol

from config.settings import settings

if TYPE_CHECKING:
    from transcription.azure_speech import AzureSpeechClient
    from transcription.chunk_manager import ChunkManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AudioFrame:
    """Single PCM audio frame received from Teams RTP."""

    data: bytes
    timestamp: float  # epoch seconds
    sequence_number: int
    sample_rate: int = 16_000
    channels: int = 1
    bits_per_sample: int = 16


@dataclass
class StreamStats:
    """Counters for monitoring the audio stream."""

    frames_received: int = 0
    frames_sent_to_speech: int = 0
    frames_sent_to_buffer: int = 0
    bytes_received: int = 0
    errors: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    @property
    def frames_per_second(self) -> float:
        elapsed = self.elapsed_seconds
        return self.frames_received / elapsed if elapsed > 0 else 0.0


# ---------------------------------------------------------------------------
# Protocol for downstream consumers (dependency injection)
# ---------------------------------------------------------------------------

class AudioConsumer(Protocol):
    """Protocol for anything that receives audio frames."""

    async def receive_frame(self, frame: AudioFrame) -> None: ...


# ---------------------------------------------------------------------------
# AudioStreamHandler
# ---------------------------------------------------------------------------

class AudioStreamHandler:
    """
    Receives raw PCM audio from Teams RTP and fans out to two consumers:

    1. **Real-time path** — Azure Speech client for live transcription.
    2. **Batch path** — ChunkManager buffer for Whisper reprocessing.

    Both paths run as fire-and-forget async tasks so the main audio loop
    is *never* blocked.
    """

    def __init__(
        self,
        speech_client: AudioConsumer | None = None,
        chunk_manager: AudioConsumer | None = None,
        *,
        max_pending_tasks: int = 1000,
    ) -> None:
        self._speech_client = speech_client
        self._chunk_manager = chunk_manager
        self._max_pending_tasks = max_pending_tasks

        self._sequence: int = 0
        self._running: bool = False
        self._stats = StreamStats()

        # Track pending tasks to avoid unbounded growth
        self._pending_tasks: set[asyncio.Task[None]] = set()

        logger.info(
            "AudioStreamHandler initialised — speech_client=%s, chunk_manager=%s",
            type(speech_client).__name__ if speech_client else "None",
            type(chunk_manager).__name__ if chunk_manager else "None",
        )

    # ---------- lifecycle ----------

    async def start(self) -> None:
        """Mark the handler as active and reset counters."""
        self._running = True
        self._sequence = 0
        self._stats = StreamStats()
        logger.info("AudioStreamHandler started — ready to receive frames")

    async def stop(self) -> None:
        """
        Stop receiving frames, flush the chunk buffer, and wait for
        all pending fan-out tasks to complete.
        """
        self._running = False
        logger.info(
            "AudioStreamHandler stopping — %d pending tasks, %d frames received",
            len(self._pending_tasks),
            self._stats.frames_received,
        )

        # Signal the chunk manager that the meeting has ended
        if self._chunk_manager and hasattr(self._chunk_manager, "flush"):
            try:
                await self._chunk_manager.flush()  # type: ignore[attr-defined]
                logger.info("Final chunk buffer flushed")
            except Exception:
                logger.exception("Error flushing final chunk buffer")

        # Wait for all in-flight tasks
        if self._pending_tasks:
            await asyncio.gather(*self._pending_tasks, return_exceptions=True)
            self._pending_tasks.clear()

        logger.info(
            "AudioStreamHandler stopped — stats: %s",
            self._stats,
        )

    # ---------- main entry point ----------

    async def receive_audio_frame(self, raw_data: bytes) -> None:
        """
        Process a single raw PCM audio frame.

        Called by the Teams media platform for every audio packet.

        Args:
            raw_data: Raw 16-bit PCM audio bytes.
        """
        if not self._running:
            logger.warning("Frame received while handler is stopped — ignoring")
            return

        self._sequence += 1
        frame = AudioFrame(
            data=raw_data,
            timestamp=time.time(),
            sequence_number=self._sequence,
            sample_rate=settings.audio.sample_rate,
            channels=settings.audio.channels,
        )

        self._stats.frames_received += 1
        self._stats.bytes_received += len(raw_data)

        # Fan out to both consumers without blocking
        self._dispatch_to_speech(frame)
        self._dispatch_to_buffer(frame)

        # Periodic logging (every 500 frames ≈ every ~16 s at 20 ms frames)
        if self._stats.frames_received % 500 == 0:
            logger.info(
                "Stream stats — frames=%d, bytes=%d, fps=%.1f, pending_tasks=%d",
                self._stats.frames_received,
                self._stats.bytes_received,
                self._stats.frames_per_second,
                len(self._pending_tasks),
            )

    # ---------- fan-out helpers ----------

    def _dispatch_to_speech(self, frame: AudioFrame) -> None:
        """Fire-and-forget the frame to the real-time speech client."""
        if self._speech_client is None:
            return
        task = asyncio.create_task(
            self._safe_send(self._speech_client, frame, "speech"),
        )
        self._track_task(task)
        self._stats.frames_sent_to_speech += 1

    def _dispatch_to_buffer(self, frame: AudioFrame) -> None:
        """Fire-and-forget the frame to the chunk buffer."""
        if self._chunk_manager is None:
            return
        task = asyncio.create_task(
            self._safe_send(self._chunk_manager, frame, "buffer"),
        )
        self._track_task(task)
        self._stats.frames_sent_to_buffer += 1

    async def _safe_send(
        self,
        consumer: AudioConsumer,
        frame: AudioFrame,
        label: str,
    ) -> None:
        """Send a frame to a consumer, catching and logging any errors."""
        try:
            await consumer.receive_frame(frame)
        except Exception:
            self._stats.errors += 1
            logger.exception(
                "Error sending frame #%d to %s consumer",
                frame.sequence_number,
                label,
            )

    def _track_task(self, task: asyncio.Task[None]) -> None:
        """Track an async task and auto-discard it on completion."""
        if len(self._pending_tasks) >= self._max_pending_tasks:
            logger.warning(
                "Pending task limit reached (%d) — dropping oldest",
                self._max_pending_tasks,
            )
            # Remove a finished task if any; otherwise skip tracking
            done = {t for t in self._pending_tasks if t.done()}
            self._pending_tasks -= done

        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)

    # ---------- accessors ----------

    @property
    def stats(self) -> StreamStats:
        """Current stream statistics."""
        return self._stats

    @property
    def is_running(self) -> bool:
        """Whether the handler is actively receiving frames."""
        return self._running


# ---------------------------------------------------------------------------
# Unit test stub
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    class _MockConsumer:
        """Minimal consumer for testing."""

        def __init__(self) -> None:
            self.frames: list[AudioFrame] = []

        async def receive_frame(self, frame: AudioFrame) -> None:
            self.frames.append(frame)

    async def _test() -> None:
        speech = _MockConsumer()
        buffer = _MockConsumer()
        handler = AudioStreamHandler(speech_client=speech, chunk_manager=buffer)

        await handler.start()
        for i in range(10):
            await handler.receive_audio_frame(b"\x00" * 640)  # 20 ms @ 16 kHz
        await handler.stop()

        assert len(speech.frames) == 10, f"Expected 10, got {len(speech.frames)}"
        assert len(buffer.frames) == 10, f"Expected 10, got {len(buffer.frames)}"
        print("✓ AudioStreamHandler test passed")

    asyncio.run(_test())
