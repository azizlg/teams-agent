"""
chunk_manager.py — Audio buffering and chunking logic.

ChunkManager accumulates PCM audio frames into configurable-duration chunks
(default 5 minutes), prefers silence boundaries for cuts, saves completed
chunks as .wav files to Azure Blob Storage, and pushes metadata to the
Redis queue for Whisper workers.
"""

from __future__ import annotations

import asyncio
import io
import logging
import struct
import time
import uuid
import wave
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel

from config.settings import settings

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class ChunkMetadata(BaseModel):
    """Metadata pushed to Redis for each completed chunk."""

    chunk_id: str
    meeting_id: str
    sequence_number: int
    blob_url: str
    duration_seconds: float
    timestamp: float
    sample_rate: int = 16_000
    channels: int = 1


@dataclass(frozen=True)
class SilenceDetectionConfig:
    """Parameters for silence boundary detection."""

    threshold_db: float = -40.0
    min_duration_ms: int = 500


# ---------------------------------------------------------------------------
# Protocols for injected dependencies
# ---------------------------------------------------------------------------

class BlobUploader(Protocol):
    """Uploads a blob and returns its URL."""

    async def upload(self, data: bytes, blob_name: str) -> str: ...


class QueuePublisher(Protocol):
    """Pushes chunk metadata to the processing queue."""

    async def push_chunk(self, metadata: ChunkMetadata) -> None: ...


# ---------------------------------------------------------------------------
# ChunkManager
# ---------------------------------------------------------------------------

class ChunkManager:
    """
    Manages the audio buffer for Whisper reprocessing.

    Accumulates PCM frames, detects silence boundaries, and flushes
    completed chunks to Blob Storage + Redis queue.

    Implements the ``AudioConsumer`` protocol so AudioStreamHandler
    can send frames to it via ``receive_frame()``.
    """

    def __init__(
        self,
        meeting_id: str,
        blob_uploader: BlobUploader | None = None,
        queue_publisher: QueuePublisher | None = None,
        *,
        chunk_duration_seconds: int | None = None,
        silence_config: SilenceDetectionConfig | None = None,
    ) -> None:
        self._meeting_id = meeting_id
        self._blob_uploader = blob_uploader
        self._queue_publisher = queue_publisher

        self._chunk_duration = chunk_duration_seconds or settings.audio.chunk_duration_seconds
        self._silence = silence_config or SilenceDetectionConfig(
            threshold_db=settings.audio.silence_threshold_db,
            min_duration_ms=settings.audio.silence_min_duration_ms,
        )

        # Buffer state
        self._buffer: bytearray = bytearray()
        self._buffer_start_time: float = time.time()
        self._sequence_number: int = 0

        # Silence tracking
        self._silence_start: float | None = None
        self._last_silence_offset: int | None = None  # byte offset of best cut

        # Bytes per second for timing calculations
        self._bps: int = settings.audio.bytes_per_second

        # Lock to prevent concurrent flush
        self._flush_lock = asyncio.Lock()

        logger.info(
            "ChunkManager initialised — meeting=%s, chunk_duration=%ds",
            meeting_id,
            self._chunk_duration,
        )

    # ---------- AudioConsumer protocol ----------

    async def receive_frame(self, frame) -> None:
        """
        Receive a PCM audio frame and append it to the buffer.

        Triggers a chunk flush if the buffer exceeds the configured duration
        or a suitable silence boundary is found after the minimum duration.
        """
        self._buffer.extend(frame.data)

        # Analyse frame for silence
        self._update_silence_tracking(frame.data)

        # Check if we should flush
        buffer_duration = len(self._buffer) / self._bps

        if buffer_duration >= self._chunk_duration:
            # Hard limit reached — cut at last silence point or right here
            await self._flush_chunk()
        elif buffer_duration >= self._chunk_duration * 0.9:
            # Within 90% of limit — prefer silence boundary
            if self._last_silence_offset is not None:
                logger.debug(
                    "Silence-boundary cut at %.1f s (buffer=%.1f s)",
                    self._last_silence_offset / self._bps,
                    buffer_duration,
                )
                await self._flush_chunk(cut_at=self._last_silence_offset)

    # ---------- silence detection ----------

    def _update_silence_tracking(self, pcm_data: bytes) -> None:
        """Track silence runs to find optimal cut boundaries."""
        rms = self._compute_rms(pcm_data)
        rms_db = self._amplitude_to_db(rms)

        if rms_db < self._silence.threshold_db:
            if self._silence_start is None:
                self._silence_start = time.time()
            else:
                silence_duration_ms = (time.time() - self._silence_start) * 1000
                if silence_duration_ms >= self._silence.min_duration_ms:
                    # Good silence region — record as potential cut point
                    self._last_silence_offset = len(self._buffer) - len(pcm_data)
        else:
            self._silence_start = None

    @staticmethod
    def _compute_rms(pcm_data: bytes) -> float:
        """Compute RMS amplitude of 16-bit PCM data."""
        if len(pcm_data) < 2:
            return 0.0
        n_samples = len(pcm_data) // 2
        samples = struct.unpack(f"<{n_samples}h", pcm_data[: n_samples * 2])
        sum_sq = sum(s * s for s in samples)
        return (sum_sq / n_samples) ** 0.5

    @staticmethod
    def _amplitude_to_db(amplitude: float) -> float:
        """Convert linear amplitude to decibels."""
        import math
        if amplitude <= 0:
            return -100.0
        return 20.0 * math.log10(amplitude / 32768.0)

    # ---------- chunk flush ----------

    async def _flush_chunk(self, *, cut_at: int | None = None) -> None:
        """Flush the current buffer as a completed chunk."""
        async with self._flush_lock:
            if not self._buffer:
                return

            # Determine cut point
            if cut_at is not None and 0 < cut_at < len(self._buffer):
                chunk_data = bytes(self._buffer[:cut_at])
                remaining = bytes(self._buffer[cut_at:])
            else:
                chunk_data = bytes(self._buffer)
                remaining = b""

            self._sequence_number += 1
            chunk_id = f"{self._meeting_id}_{self._sequence_number:04d}_{uuid.uuid4().hex[:8]}"
            duration = len(chunk_data) / self._bps

            logger.info(
                "Flushing chunk #%d — size=%d bytes, duration=%.1f s, meeting=%s",
                self._sequence_number,
                len(chunk_data),
                duration,
                self._meeting_id,
            )

            # Build WAV file in memory
            wav_bytes = self._pcm_to_wav(chunk_data)
            blob_name = f"{self._meeting_id}/{chunk_id}.wav"

            # Upload to Blob Storage
            blob_url = ""
            if self._blob_uploader:
                try:
                    blob_url = await self._blob_uploader.upload(wav_bytes, blob_name)
                    logger.info("Chunk %s uploaded → %s", chunk_id, blob_url)
                except Exception:
                    logger.exception("Failed to upload chunk %s", chunk_id)

            # Push metadata to Redis queue
            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                meeting_id=self._meeting_id,
                sequence_number=self._sequence_number,
                blob_url=blob_url,
                duration_seconds=duration,
                timestamp=self._buffer_start_time,
                sample_rate=settings.audio.sample_rate,
                channels=settings.audio.channels,
            )

            if self._queue_publisher:
                try:
                    await self._queue_publisher.push_chunk(metadata)
                    logger.info("Chunk %s queued for Whisper processing", chunk_id)
                except Exception:
                    logger.exception("Failed to queue chunk %s", chunk_id)

            # Reset buffer with any remaining data
            self._buffer = bytearray(remaining)
            self._buffer_start_time = time.time()
            self._silence_start = None
            self._last_silence_offset = None

    async def flush(self) -> None:
        """
        Force-flush any remaining data in the buffer.

        Called when a meeting ends to process the final partial chunk.
        """
        if self._buffer:
            logger.info(
                "Force-flushing remaining buffer — %d bytes (%.1f s)",
                len(self._buffer),
                len(self._buffer) / self._bps,
            )
            await self._flush_chunk()

    # ---------- WAV encoding ----------

    def _pcm_to_wav(self, pcm_data: bytes) -> bytes:
        """Convert raw PCM bytes to a WAV file in memory."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(settings.audio.channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(settings.audio.sample_rate)
            wf.writeframes(pcm_data)
        return buf.getvalue()

    # ---------- accessors ----------

    @property
    def buffer_duration_seconds(self) -> float:
        """Current buffer duration in seconds."""
        return len(self._buffer) / self._bps if self._bps else 0.0

    @property
    def sequence_number(self) -> int:
        """Number of chunks flushed so far."""
        return self._sequence_number


# ---------------------------------------------------------------------------
# Unit test stub
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio
    from dataclasses import dataclass as dc

    @dc(frozen=True)
    class _FakeFrame:
        data: bytes
        timestamp: float = 0.0
        sequence_number: int = 0

    class _FakeUploader:
        def __init__(self) -> None:
            self.uploads: list[str] = []

        async def upload(self, data: bytes, blob_name: str) -> str:
            self.uploads.append(blob_name)
            return f"https://blob.example.com/{blob_name}"

    class _FakeQueue:
        def __init__(self) -> None:
            self.chunks: list[ChunkMetadata] = []

        async def push_chunk(self, metadata: ChunkMetadata) -> None:
            self.chunks.append(metadata)

    async def _test() -> None:
        uploader = _FakeUploader()
        queue = _FakeQueue()
        mgr = ChunkManager(
            meeting_id="test-meeting",
            blob_uploader=uploader,
            queue_publisher=queue,
            chunk_duration_seconds=1,  # 1 second for fast test
        )

        # Send ~1.5 seconds of silence
        frame_bytes = b"\x00" * 640  # 20 ms
        for i in range(75):
            await mgr.receive_frame(_FakeFrame(data=frame_bytes, sequence_number=i))

        # Force flush remaining
        await mgr.flush()

        assert len(uploader.uploads) > 0, "Expected at least one upload"
        assert len(queue.chunks) > 0, "Expected at least one chunk"
        print(f"✓ ChunkManager test passed — {len(queue.chunks)} chunk(s) created")

    asyncio.run(_test())
