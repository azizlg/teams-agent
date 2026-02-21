"""
diarization.py — pyannote speaker diarization.

Identifies who is speaking at each point in the audio using the
pyannote.audio speaker diarization pipeline. Maps speaker embeddings
to participant identities when possible.
"""

from __future__ import annotations

import asyncio
import io
import logging
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from config.settings import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class SpeakerSegment(BaseModel):
    """A segment attributed to a specific speaker."""

    speaker_id: str  # e.g. "SPEAKER_00"
    start: float  # seconds
    end: float  # seconds
    confidence: float = 0.0


class DiarizationResult(BaseModel):
    """Full diarization result for an audio chunk."""

    chunk_id: str
    meeting_id: str
    segments: list[SpeakerSegment]
    num_speakers: int
    processing_time_seconds: float = 0.0


@dataclass
class SpeakerProfile:
    """
    Maps a pyannote speaker label to a real participant.

    Built up over the course of a meeting as we match
    speaker embeddings against known participant voice profiles.
    """

    speaker_label: str  # pyannote label, e.g. "SPEAKER_00"
    participant_id: str | None = None  # matched Teams participant ID
    participant_name: str | None = None
    embedding: Any = None  # speaker embedding vector
    total_speaking_time: float = 0.0  # seconds


# ---------------------------------------------------------------------------
# SpeakerDiarizer
# ---------------------------------------------------------------------------

class SpeakerDiarizer:
    """
    Speaker diarization engine using pyannote.audio.

    Processes audio chunks to identify speaker turns and
    maintain speaker profiles across the meeting.

    Usage:
        diarizer = SpeakerDiarizer()
        await diarizer.load_model()
        result = await diarizer.diarize(audio_bytes, chunk_id, meeting_id)
    """

    def __init__(
        self,
        *,
        auth_token: str | None = None,
        num_speakers: int | None = None,
        min_speakers: int = 1,
        max_speakers: int = 10,
    ) -> None:
        self._auth_token = auth_token or settings.pyannote.auth_token
        self._num_speakers = num_speakers
        self._min_speakers = min_speakers
        self._max_speakers = max_speakers

        self._pipeline: Any = None  # pyannote Pipeline object
        self._loaded: bool = False

        # Speaker profiles built across the meeting
        self._speaker_profiles: dict[str, SpeakerProfile] = {}

        logger.info(
            "SpeakerDiarizer initialised — speakers=(%d–%d)",
            self._min_speakers,
            self._max_speakers,
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    async def load_model(self) -> None:
        """
        Load the pyannote speaker diarization pipeline.

        Runs in a thread executor since model loading is CPU-bound.
        """
        if self._loaded:
            logger.debug("Diarization model already loaded")
            return

        logger.info("Loading pyannote speaker diarization pipeline...")
        loop = asyncio.get_running_loop()
        self._pipeline = await loop.run_in_executor(None, self._load_pipeline)
        self._loaded = True
        logger.info("pyannote pipeline loaded and ready")

    def _load_pipeline(self) -> Any:
        """Synchronous pipeline loading (runs in executor)."""
        from pyannote.audio import Pipeline

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=self._auth_token,
        )

        # Move to GPU if available
        try:
            import torch
            if torch.cuda.is_available():
                pipeline.to(torch.device("cuda"))
                logger.info("pyannote pipeline moved to GPU")
        except ImportError:
            logger.debug("PyTorch not available for GPU acceleration")

        return pipeline

    # ------------------------------------------------------------------
    # Diarization
    # ------------------------------------------------------------------

    async def diarize(
        self,
        audio_bytes: bytes,
        chunk_id: str,
        meeting_id: str,
    ) -> DiarizationResult:
        """
        Run speaker diarization on an audio chunk.

        Args:
            audio_bytes: Raw WAV file bytes.
            chunk_id: Identifier for this audio chunk.
            meeting_id: Parent meeting identifier.

        Returns:
            DiarizationResult with speaker segments.
        """
        if not self._loaded or self._pipeline is None:
            raise RuntimeError("Pipeline not loaded — call load_model() first")

        import time as _time
        start_time = _time.time()

        logger.info("Running diarization on chunk %s", chunk_id)

        loop = asyncio.get_running_loop()

        # Write audio to temp file (pyannote needs a file path)
        tmp_path = Path(tempfile.mktemp(suffix=".wav"))
        try:
            tmp_path.write_bytes(audio_bytes)

            # Run diarization in thread executor (CPU/GPU-bound)
            diarization = await loop.run_in_executor(
                None,
                lambda: self._run_pipeline(str(tmp_path)),
            )

            # Convert pyannote output to our data model
            segments: list[SpeakerSegment] = []
            speakers_seen: set[str] = set()

            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segment = SpeakerSegment(
                    speaker_id=speaker,
                    start=turn.start,
                    end=turn.end,
                )
                segments.append(segment)
                speakers_seen.add(speaker)

                # Update speaker profile
                self._update_speaker_profile(speaker, turn.end - turn.start)

            processing_time = _time.time() - start_time

            result = DiarizationResult(
                chunk_id=chunk_id,
                meeting_id=meeting_id,
                segments=segments,
                num_speakers=len(speakers_seen),
                processing_time_seconds=processing_time,
            )

            logger.info(
                "Diarization complete for chunk %s — %d speakers, %d segments, %.1f s",
                chunk_id,
                result.num_speakers,
                len(segments),
                processing_time,
            )

            return result

        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    def _run_pipeline(self, audio_path: str) -> Any:
        """Run the pyannote pipeline (synchronous, runs in executor)."""
        params: dict[str, Any] = {}

        if self._num_speakers is not None:
            params["num_speakers"] = self._num_speakers
        else:
            params["min_speakers"] = self._min_speakers
            params["max_speakers"] = self._max_speakers

        return self._pipeline(audio_path, **params)

    # ------------------------------------------------------------------
    # Speaker profile management
    # ------------------------------------------------------------------

    def _update_speaker_profile(self, speaker_label: str, duration: float) -> None:
        """Update or create a speaker profile."""
        if speaker_label not in self._speaker_profiles:
            self._speaker_profiles[speaker_label] = SpeakerProfile(
                speaker_label=speaker_label,
            )
        self._speaker_profiles[speaker_label].total_speaking_time += duration

    def map_speaker_to_participant(
        self,
        speaker_label: str,
        participant_id: str,
        participant_name: str,
    ) -> None:
        """
        Map a pyannote speaker label to a known meeting participant.

        Call this when the identity of a speaker has been determined
        (e.g., via manual mapping or voice profile matching).
        """
        if speaker_label not in self._speaker_profiles:
            self._speaker_profiles[speaker_label] = SpeakerProfile(
                speaker_label=speaker_label,
            )

        profile = self._speaker_profiles[speaker_label]
        profile.participant_id = participant_id
        profile.participant_name = participant_name

        logger.info(
            "Mapped speaker %s → %s (%s)",
            speaker_label,
            participant_name,
            participant_id,
        )

    def resolve_speaker(self, speaker_label: str) -> str:
        """
        Resolve a speaker label to a participant name if mapped.

        Returns the participant name, or the raw label if not mapped.
        """
        profile = self._speaker_profiles.get(speaker_label)
        if profile and profile.participant_name:
            return profile.participant_name
        return speaker_label

    def get_speaker_summary(self) -> list[dict[str, Any]]:
        """Get a summary of all speakers and their speaking time."""
        return [
            {
                "speaker_label": p.speaker_label,
                "participant_id": p.participant_id,
                "participant_name": p.participant_name or p.speaker_label,
                "total_speaking_time": round(p.total_speaking_time, 1),
            }
            for p in sorted(
                self._speaker_profiles.values(),
                key=lambda x: x.total_speaking_time,
                reverse=True,
            )
        ]

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def speaker_count(self) -> int:
        return len(self._speaker_profiles)


# ---------------------------------------------------------------------------
# Alignment utility
# ---------------------------------------------------------------------------

def align_transcript_with_speakers(
    transcript_segments: list[dict[str, Any]],
    diarization_segments: list[SpeakerSegment],
) -> list[dict[str, Any]]:
    """
    Align transcript text segments with speaker diarization segments.

    For each transcript segment, finds the overlapping speaker segment
    and assigns the speaker_id. Uses maximum overlap strategy.

    Args:
        transcript_segments: List of dicts with 'start', 'end', 'text' keys.
        diarization_segments: List of SpeakerSegment instances.

    Returns:
        The transcript_segments list with 'speaker_id' field populated.
    """
    for tseg in transcript_segments:
        t_start = tseg.get("start", 0.0)
        t_end = tseg.get("end", 0.0)

        best_speaker = None
        best_overlap = 0.0

        for dseg in diarization_segments:
            # Calculate overlap
            overlap_start = max(t_start, dseg.start)
            overlap_end = min(t_end, dseg.end)
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = dseg.speaker_id

        tseg["speaker_id"] = best_speaker

    return transcript_segments


# ---------------------------------------------------------------------------
# Unit test stub
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    async def _test() -> None:
        diarizer = SpeakerDiarizer()
        assert not diarizer.is_loaded
        assert diarizer.speaker_count == 0

        # Test speaker mapping
        diarizer.map_speaker_to_participant("SPEAKER_00", "user-1", "Alice")
        diarizer.map_speaker_to_participant("SPEAKER_01", "user-2", "Bob")
        assert diarizer.resolve_speaker("SPEAKER_00") == "Alice"
        assert diarizer.resolve_speaker("SPEAKER_02") == "SPEAKER_02"

        # Test alignment
        transcript = [
            {"start": 0.0, "end": 2.5, "text": "Hello everyone"},
            {"start": 3.0, "end": 5.0, "text": "Hi there"},
        ]
        speakers = [
            SpeakerSegment(speaker_id="SPEAKER_00", start=0.0, end=3.0),
            SpeakerSegment(speaker_id="SPEAKER_01", start=3.0, end=6.0),
        ]
        aligned = align_transcript_with_speakers(transcript, speakers)
        assert aligned[0]["speaker_id"] == "SPEAKER_00"
        assert aligned[1]["speaker_id"] == "SPEAKER_01"

        print("✓ SpeakerDiarizer test passed")
        print(f"  Speaker summary: {diarizer.get_speaker_summary()}")

    import asyncio
    asyncio.run(_test())
