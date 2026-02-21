"""
whisper_realtime.py — Near-real-time Whisper transcription.

Replaces Azure Speech for real-time transcription. Buffers audio chunks
(default: ~10 seconds), runs faster-whisper, and emits transcript segments
via the same callback interface.
"""

from __future__ import annotations

import asyncio
import io
import logging
import struct
import tempfile
import time
import wave
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Awaitable

from config.settings import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TranscriptSegment:
    """A single recognised transcript segment."""

    text: str
    language: str
    timestamp: float  # epoch seconds
    offset_ms: int  # offset within the audio stream in milliseconds
    duration_ms: int  # duration of the segment in milliseconds
    confidence: float  # 0.0-1.0
    speaker_id: str | None = None  # placeholder for diarization


# Type alias for the callback
SegmentCallback = Callable[[TranscriptSegment], Awaitable[None]]


# ---------------------------------------------------------------------------
# WhisperRealtimeTranscriber
# ---------------------------------------------------------------------------

class WhisperRealtimeTranscriber:
    """
    Near-real-time speech recognition using faster-whisper.

    Replaces AzureSpeechClient for on-device transcription. Buffers
    incoming audio frames and periodically processes them with Whisper.

    Features:
    - Auto language detection
    - Configurable chunk duration (default 10s)
    - Same callback interface as the old AzureSpeechClient
    - Push-stream audio input

    Usage:
        async def on_segment(seg):
            print(f"[{seg.language}] {seg.text}")

        transcriber = WhisperRealtimeTranscriber(on_segment=on_segment)
        await transcriber.start()
        await transcriber.receive_frame(audio_frame)
        ...
        await transcriber.stop()
    """

    def __init__(
        self,
        on_segment: SegmentCallback | None = None,
        *,
        model_size: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
        chunk_seconds: float | None = None,
    ) -> None:
        self._on_segment = on_segment
        self._model_size = model_size or settings.whisper.model_size
        self._device = device or settings.whisper.device
        self._compute_type = compute_type or settings.whisper.compute_type
        self._chunk_seconds = chunk_seconds or settings.whisper.realtime_chunk_seconds

        self._sample_rate = settings.audio.sample_rate
        self._channels = settings.audio.channels
        self._bytes_per_second = self._sample_rate * self._channels * 2  # 16-bit

        # Buffer for incoming audio
        self._audio_buffer = bytearray()
        self._buffer_lock = asyncio.Lock()
        self._chunk_size_bytes = int(self._bytes_per_second * self._chunk_seconds)

        # Whisper model (loaded lazily)
        self._model: Any = None
        self._running: bool = False
        self._process_task: asyncio.Task | None = None

        # Timing
        self._start_time: float = 0.0
        self._total_audio_seconds: float = 0.0

        # Stats
        self._segments_recognised: int = 0

        logger.info(
            "WhisperRealtimeTranscriber initialised — model=%s, chunk=%.1fs",
            self._model_size,
            self._chunk_seconds,
        )

    # ---------- lifecycle ----------

    async def start(self) -> None:
        """Load the Whisper model and start the processing loop."""
        if self._running:
            logger.warning("WhisperRealtimeTranscriber already running")
            return

        logger.info("Loading Whisper model '%s' for real-time...", self._model_size)

        loop = asyncio.get_running_loop()
        self._model = await loop.run_in_executor(None, self._load_model)
        self._running = True
        self._start_time = time.time()

        # Start background processing loop
        self._process_task = asyncio.create_task(self._processing_loop())

        logger.info("WhisperRealtimeTranscriber started — model loaded")

    def _load_model(self):
        """Load the Whisper model (synchronous, runs in executor)."""
        from faster_whisper import WhisperModel

        device = self._device
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        compute = self._compute_type if device != "cpu" else "int8"

        logger.info(
            "Whisper realtime: device=%s, compute=%s",
            device,
            compute,
        )

        return WhisperModel(
            self._model_size,
            device=device,
            compute_type=compute,
        )

    async def stop(self) -> None:
        """Stop the processing loop, flush remaining audio."""
        if not self._running:
            return

        self._running = False

        # Process any remaining audio in the buffer
        await self._flush_buffer()

        # Cancel the background task
        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass

        logger.info(
            "WhisperRealtimeTranscriber stopped — %d segments, %.1f s audio",
            self._segments_recognised,
            self._total_audio_seconds,
        )

    # ---------- AudioConsumer protocol ----------

    async def receive_frame(self, frame) -> None:
        """
        Receive a PCM audio frame and buffer it.

        Implements the same interface as the old AzureSpeechClient
        so AudioStreamHandler can send frames directly.
        """
        async with self._buffer_lock:
            self._audio_buffer.extend(frame.data)

    # ---------- processing loop ----------

    async def _processing_loop(self) -> None:
        """Background loop that processes audio chunks periodically."""
        logger.info("Realtime processing loop started (chunk=%.1fs)", self._chunk_seconds)

        while self._running:
            await asyncio.sleep(self._chunk_seconds)

            async with self._buffer_lock:
                if len(self._audio_buffer) < self._chunk_size_bytes // 2:
                    # Not enough audio yet — skip this cycle
                    continue

                # Extract the chunk
                chunk_data = bytes(self._audio_buffer[:self._chunk_size_bytes])
                self._audio_buffer = self._audio_buffer[self._chunk_size_bytes:]

            if chunk_data:
                await self._process_chunk(chunk_data)

    async def _flush_buffer(self) -> None:
        """Process any remaining audio in the buffer."""
        async with self._buffer_lock:
            if len(self._audio_buffer) > self._bytes_per_second:  # at least 1 second
                chunk_data = bytes(self._audio_buffer)
                self._audio_buffer.clear()
                await self._process_chunk(chunk_data)
            else:
                self._audio_buffer.clear()

    async def _process_chunk(self, pcm_data: bytes) -> None:
        """Transcribe a chunk of PCM audio with Whisper."""
        if not self._model:
            return

        chunk_duration = len(pcm_data) / self._bytes_per_second
        self._total_audio_seconds += chunk_duration

        loop = asyncio.get_running_loop()

        try:
            # Write to temp WAV file (faster-whisper needs file path)
            tmp_path = Path(tempfile.mktemp(suffix=".wav"))
            try:
                self._write_wav(tmp_path, pcm_data)

                # Run transcription in executor
                segments_raw, info = await loop.run_in_executor(
                    None,
                    lambda: self._model.transcribe(
                        str(tmp_path),
                        beam_size=3,  # faster than default 5
                        best_of=1,
                        language=None,  # auto-detect
                        vad_filter=True,
                        vad_parameters=dict(
                            min_silence_duration_ms=300,
                            speech_pad_ms=150,
                        ),
                    ),
                )

                segments_list = await loop.run_in_executor(
                    None,
                    lambda: list(segments_raw),
                )

                # Calculate the stream offset for this chunk
                stream_offset_ms = int(
                    (self._total_audio_seconds - chunk_duration) * 1000
                )

                # Emit segments
                for seg in segments_list:
                    text = seg.text.strip()
                    if not text:
                        continue

                    import math
                    confidence = max(0.0, min(1.0, math.exp(seg.avg_logprob)))

                    segment = TranscriptSegment(
                        text=text,
                        language=info.language,
                        timestamp=time.time(),
                        offset_ms=stream_offset_ms + int(seg.start * 1000),
                        duration_ms=int((seg.end - seg.start) * 1000),
                        confidence=confidence,
                    )

                    self._segments_recognised += 1

                    logger.info(
                        "Recognised [%s] (%.0f%%): %s",
                        segment.language,
                        segment.confidence * 100,
                        segment.text[:80],
                    )

                    # Call async callback
                    if self._on_segment:
                        try:
                            await self._on_segment(segment)
                        except Exception:
                            logger.exception("Error in segment callback")

            finally:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass

        except Exception:
            logger.exception("Error processing audio chunk")

    def _write_wav(self, path: Path, pcm_data: bytes) -> None:
        """Write raw PCM data to a WAV file."""
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(self._channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self._sample_rate)
            wf.writeframes(pcm_data)

    # ---------- accessors ----------

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def segments_recognised(self) -> int:
        return self._segments_recognised


# ---------------------------------------------------------------------------
# Unit test stub
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    async def _test() -> None:
        transcriber = WhisperRealtimeTranscriber()
        assert not transcriber.is_running
        assert transcriber.segments_recognised == 0
        print("WhisperRealtimeTranscriber initialisation test passed")
        print("  (Actual transcription requires faster-whisper installed)")

    import asyncio
    asyncio.run(_test())
