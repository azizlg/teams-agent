"""
whisper_worker.py — faster-whisper processing worker.

WhisperWorker runs as a standalone async process, polling Redis for new
audio chunks, downloading them from Azure Blob Storage, transcribing with
faster-whisper large-v3, and saving cleaned results to PostgreSQL.
Falls back to the Azure Speech transcript if Whisper processing fails.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from config.settings import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class WhisperSegment(BaseModel):
    """A single transcript segment produced by Whisper."""

    text: str
    start: float  # seconds within chunk
    end: float  # seconds within chunk
    language: str
    confidence: float  # avg_logprob converted to rough confidence


class ChunkTranscript(BaseModel):
    """Full transcript result for a single audio chunk."""

    chunk_id: str
    meeting_id: str
    sequence_number: int
    segments: list[WhisperSegment]
    language: str
    processing_time_seconds: float
    source: str = "whisper"  # "whisper" or "azure_fallback"


# ---------------------------------------------------------------------------
# Protocols for injected dependencies
# ---------------------------------------------------------------------------

class BlobDownloader(Protocol):
    """Downloads a blob by URL and returns its bytes."""

    async def download(self, blob_url: str) -> bytes: ...


class TranscriptStore(Protocol):
    """Persists a chunk transcript to the database."""

    async def save_transcript(self, transcript: ChunkTranscript) -> None: ...


class ChunkEvent(Protocol):
    """Emits events about chunk processing status."""

    async def emit(self, event_type: str, data: dict) -> None: ...


# ---------------------------------------------------------------------------
# WhisperWorker
# ---------------------------------------------------------------------------

class WhisperWorker:
    """
    Standalone async worker that consumes audio chunks from Redis,
    transcribes them with faster-whisper, and persists results.

    Designed to run as a separate process (``python -m transcription.whisper_worker``).
    """

    def __init__(
        self,
        blob_downloader: BlobDownloader | None = None,
        transcript_store: TranscriptStore | None = None,
        event_emitter: ChunkEvent | None = None,
        *,
        model_size: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
    ) -> None:
        self._blob_downloader = blob_downloader
        self._transcript_store = transcript_store
        self._event_emitter = event_emitter

        self._model_size = model_size or settings.whisper.model_size
        self._device = device or settings.whisper.device
        self._compute_type = compute_type or settings.whisper.compute_type

        self._model = None  # Loaded lazily on startup
        self._running: bool = False
        self._chunks_processed: int = 0
        self._errors: int = 0

        logger.info(
            "WhisperWorker initialised — model=%s, device=%s, compute=%s",
            self._model_size,
            self._device,
            self._compute_type,
        )

    # ---------- lifecycle ----------

    async def start(self) -> None:
        """Load the Whisper model into memory."""
        logger.info("Loading faster-whisper model '%s'...", self._model_size)

        loop = asyncio.get_running_loop()
        self._model = await loop.run_in_executor(None, self._load_model)
        self._running = True

        logger.info("Whisper model loaded and ready")

    def _load_model(self):
        """Synchronous model loading (runs in thread executor)."""
        from faster_whisper import WhisperModel

        device = self._device
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        logger.info("Loading Whisper on device=%s, compute_type=%s", device, self._compute_type)

        return WhisperModel(
            self._model_size,
            device=device,
            compute_type=self._compute_type if device != "cpu" else "int8",
        )

    async def stop(self) -> None:
        """Stop the worker."""
        self._running = False
        logger.info(
            "WhisperWorker stopped — processed=%d, errors=%d",
            self._chunks_processed,
            self._errors,
        )

    # ---------- chunk processing ----------

    async def process_chunk(self, chunk_metadata: dict) -> ChunkTranscript | None:
        """
        Process a single audio chunk end-to-end:
        1. Download .wav from Blob Storage
        2. Transcribe with faster-whisper
        3. Save transcript to database
        4. Emit "chunk processed" event

        Args:
            chunk_metadata: Dict with keys chunk_id, meeting_id,
                            sequence_number, blob_url, etc.

        Returns:
            The ChunkTranscript, or None if processing fails completely.
        """
        chunk_id = chunk_metadata.get("chunk_id", "unknown")
        meeting_id = chunk_metadata.get("meeting_id", "unknown")
        sequence_number = int(chunk_metadata.get("sequence_number", 0))
        blob_url = chunk_metadata.get("blob_url", "")

        logger.info(
            "Processing chunk %s (seq=%d, meeting=%s)",
            chunk_id,
            sequence_number,
            meeting_id,
        )

        start_time = time.time()

        try:
            # Step 1: Download audio
            audio_bytes = await self._download_audio(blob_url, chunk_id)
            if not audio_bytes:
                return None

            # Step 2: Transcribe with Whisper
            transcript = await self._transcribe(
                audio_bytes,
                chunk_id=chunk_id,
                meeting_id=meeting_id,
                sequence_number=sequence_number,
            )
            transcript.processing_time_seconds = time.time() - start_time

            # Step 3: Save to database
            if self._transcript_store:
                try:
                    await self._transcript_store.save_transcript(transcript)
                    logger.info("Transcript saved for chunk %s", chunk_id)
                except Exception:
                    logger.exception("Failed to save transcript for chunk %s", chunk_id)

            # Step 4: Emit event
            if self._event_emitter:
                try:
                    await self._event_emitter.emit(
                        "chunk_processed",
                        {
                            "chunk_id": chunk_id,
                            "meeting_id": meeting_id,
                            "sequence_number": sequence_number,
                            "segments_count": len(transcript.segments),
                            "language": transcript.language,
                            "processing_time": transcript.processing_time_seconds,
                        },
                    )
                except Exception:
                    logger.exception("Failed to emit event for chunk %s", chunk_id)

            self._chunks_processed += 1
            logger.info(
                "Chunk %s processed in %.1f s — %d segments, language=%s",
                chunk_id,
                transcript.processing_time_seconds,
                len(transcript.segments),
                transcript.language,
            )
            return transcript

        except Exception:
            self._errors += 1
            logger.exception("Failed to process chunk %s", chunk_id)
            return None

    async def _download_audio(self, blob_url: str, chunk_id: str) -> bytes | None:
        """Download audio from Blob Storage."""
        if not self._blob_downloader:
            logger.error("No blob downloader configured — cannot download chunk %s", chunk_id)
            return None

        try:
            audio_bytes = await self._blob_downloader.download(blob_url)
            logger.debug(
                "Downloaded chunk %s — %d bytes",
                chunk_id,
                len(audio_bytes),
            )
            return audio_bytes
        except Exception:
            logger.exception("Failed to download chunk %s from %s", chunk_id, blob_url)
            return None

    async def _transcribe(
        self,
        audio_bytes: bytes,
        *,
        chunk_id: str,
        meeting_id: str,
        sequence_number: int,
    ) -> ChunkTranscript:
        """Run Whisper transcription on audio bytes."""
        if self._model is None:
            raise RuntimeError("Whisper model not loaded — call start() first")

        loop = asyncio.get_running_loop()

        # Write audio to temp file (faster-whisper needs a file path)
        tmp_path = Path(tempfile.mktemp(suffix=".wav"))
        try:
            tmp_path.write_bytes(audio_bytes)

            # Run transcription in thread executor (CPU/GPU-bound)
            segments_raw, info = await loop.run_in_executor(
                None,
                lambda: self._model.transcribe(
                    str(tmp_path),
                    beam_size=5,
                    best_of=5,
                    language=None,  # auto-detect
                    vad_filter=True,
                    vad_parameters=dict(
                        min_silence_duration_ms=500,
                        speech_pad_ms=200,
                    ),
                ),
            )

            # Materialise segment iterator in executor too
            segments_list = await loop.run_in_executor(
                None,
                lambda: list(segments_raw),
            )

            whisper_segments = [
                WhisperSegment(
                    text=seg.text.strip(),
                    start=seg.start,
                    end=seg.end,
                    language=info.language,
                    confidence=self._logprob_to_confidence(seg.avg_logprob),
                )
                for seg in segments_list
                if seg.text.strip()
            ]

            return ChunkTranscript(
                chunk_id=chunk_id,
                meeting_id=meeting_id,
                sequence_number=sequence_number,
                segments=whisper_segments,
                language=info.language,
                processing_time_seconds=0.0,  # filled by caller
            )

        finally:
            # Cleanup temp file
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    @staticmethod
    def _logprob_to_confidence(avg_logprob: float) -> float:
        """Convert average log probability to a rough 0–1 confidence score."""
        import math
        return max(0.0, min(1.0, math.exp(avg_logprob)))

    # ---------- main consumer loop ----------

    async def run_consumer(self, redis_queue) -> None:
        """
        Main loop: continuously consume chunks from Redis and process them.

        This is the entry point when running as ``python -m transcription.whisper_worker``.
        """
        logger.info("WhisperWorker consumer loop starting...")

        async def _process_callback(chunk_metadata: dict) -> bool:
            """Process a chunk; return True if successful."""
            result = await self.process_chunk(chunk_metadata)
            return result is not None

        await redis_queue.consume_chunks(_process_callback)

    # ---------- accessors ----------

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def chunks_processed(self) -> int:
        return self._chunks_processed


# ---------------------------------------------------------------------------
# Standalone worker entry point
# ---------------------------------------------------------------------------

async def _main() -> None:
    """Entry point for ``python -m transcription.whisper_worker``."""
    import sys

    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("Starting WhisperWorker as standalone process...")

    # Import and initialise dependencies
    from storage.redis_queue import RedisQueue

    redis_queue = RedisQueue()
    await redis_queue.connect()

    worker = WhisperWorker()
    await worker.start()

    try:
        await worker.run_consumer(redis_queue)
    except KeyboardInterrupt:
        logger.info("WhisperWorker interrupted")
    finally:
        await worker.stop()
        await redis_queue.close()


if __name__ == "__main__":
    asyncio.run(_main())
