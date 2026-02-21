"""
azure_speech.py — Real-time Azure Speech client.

AzureSpeechClient wraps the Azure Cognitive Services Speech SDK for
continuous recognition with automatic language detection across 10 languages.
Each recognized segment emits text, detected language, timestamp, and confidence.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Awaitable

import azure.cognitiveservices.speech as speechsdk

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
    confidence: float  # 0.0–1.0
    speaker_id: str | None = None  # placeholder for diarization


# Type alias for the callback
SegmentCallback = Callable[[TranscriptSegment], Awaitable[None]]


# ---------------------------------------------------------------------------
# AzureSpeechClient
# ---------------------------------------------------------------------------

class AzureSpeechClient:
    """
    Continuous async speech recognition using the Azure Speech SDK.

    Features:
    - Auto language detection across 10 languages
    - Continuous recognition (not one-shot)
    - Callback on each recognised segment
    - Push-stream audio input for integration with AudioStreamHandler
    - Graceful start/stop tied to meeting lifecycle
    """

    def __init__(
        self,
        on_segment: SegmentCallback | None = None,
        *,
        speech_key: str | None = None,
        speech_region: str | None = None,
        languages: list[str] | None = None,
    ) -> None:
        self._speech_key = speech_key or settings.speech.key
        self._speech_region = speech_region or settings.speech.region
        self._languages = languages or list(settings.speech.languages)

        self._on_segment = on_segment
        self._loop: asyncio.AbstractEventLoop | None = None

        # SDK objects (initialised on start)
        self._push_stream: speechsdk.audio.PushAudioInputStream | None = None
        self._recognizer: speechsdk.SpeechRecognizer | None = None
        self._running: bool = False

        # Stats
        self._segments_recognised: int = 0
        self._errors: int = 0

        logger.info(
            "AzureSpeechClient initialised — region=%s, languages=%s",
            self._speech_region,
            self._languages,
        )

    # ---------- AudioConsumer protocol ----------

    async def receive_frame(self, frame) -> None:
        """
        Receive a PCM audio frame and push it into the SDK stream.

        Implements the AudioConsumer protocol so AudioStreamHandler
        can send frames directly.
        """
        if self._push_stream is not None:
            # push_stream.write is synchronous but very fast (buffer copy)
            self._push_stream.write(frame.data)

    # ---------- lifecycle ----------

    async def start(self) -> None:
        """
        Start continuous speech recognition.

        Sets up the push audio stream, configures language detection,
        and begins async recognition.
        """
        if self._running:
            logger.warning("AzureSpeechClient already running")
            return

        self._loop = asyncio.get_running_loop()

        # --- Audio input: push stream ---
        audio_format = speechsdk.audio.AudioStreamFormat(
            samples_per_second=settings.audio.sample_rate,
            bits_per_sample=16,
            channels=settings.audio.channels,
        )
        self._push_stream = speechsdk.audio.PushAudioInputStream(audio_format)
        audio_config = speechsdk.audio.AudioConfig(stream=self._push_stream)

        # --- Speech config ---
        speech_config = speechsdk.SpeechConfig(
            subscription=self._speech_key,
            region=self._speech_region,
        )
        speech_config.set_profanity(speechsdk.ProfanityOption.Raw)
        speech_config.output_format = speechsdk.OutputFormat.Detailed

        # --- Auto language detection ---
        auto_detect_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
            languages=self._languages,
        )

        # --- Recognizer ---
        self._recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config,
            auto_detect_source_language_config=auto_detect_config,
        )

        # Wire up callbacks
        self._recognizer.recognized.connect(self._on_recognized)
        self._recognizer.recognizing.connect(self._on_recognizing)
        self._recognizer.canceled.connect(self._on_canceled)
        self._recognizer.session_stopped.connect(self._on_session_stopped)

        # Start continuous recognition
        self._recognizer.start_continuous_recognition_async()
        self._running = True

        logger.info("AzureSpeechClient started — continuous recognition active")

    async def stop(self) -> None:
        """Stop continuous recognition and close the push stream."""
        if not self._running:
            return

        self._running = False

        if self._push_stream:
            self._push_stream.close()

        if self._recognizer:
            self._recognizer.stop_continuous_recognition_async()
            self._recognizer = None

        logger.info(
            "AzureSpeechClient stopped — %d segments, %d errors",
            self._segments_recognised,
            self._errors,
        )

    # ---------- SDK callbacks (run on SDK thread — bridge to asyncio) ----------

    def _on_recognized(self, evt: speechsdk.SpeechRecognitionEventArgs) -> None:
        """Called by SDK when a complete utterance is recognized."""
        if evt.result.reason != speechsdk.ResultReason.RecognizedSpeech:
            return

        if not evt.result.text.strip():
            return

        # Extract language and confidence from detailed results
        language = "unknown"
        confidence = 0.0

        auto_detect_result = speechsdk.AutoDetectSourceLanguageResult(evt.result)
        if auto_detect_result.language:
            language = auto_detect_result.language

        # Try to get confidence from detailed results JSON
        try:
            detailed = evt.result.properties.get(
                speechsdk.PropertyId.SpeechServiceResponse_JsonResult, ""
            )
            if detailed:
                import json
                parsed = json.loads(detailed)
                nbest = parsed.get("NBest", [])
                if nbest:
                    confidence = nbest[0].get("Confidence", 0.0)
        except Exception:
            logger.debug("Could not parse detailed results for confidence")

        segment = TranscriptSegment(
            text=evt.result.text,
            language=language,
            timestamp=time.time(),
            offset_ms=int(evt.result.offset / 10_000),  # ticks → ms
            duration_ms=int(evt.result.duration / 10_000),
            confidence=confidence,
        )

        self._segments_recognised += 1

        logger.info(
            "Recognised [%s] (%.0f%%): %s",
            segment.language,
            segment.confidence * 100,
            segment.text[:80],
        )

        # Bridge to async callback
        if self._on_segment and self._loop:
            asyncio.run_coroutine_threadsafe(
                self._on_segment(segment),
                self._loop,
            )

    def _on_recognizing(self, evt: speechsdk.SpeechRecognitionEventArgs) -> None:
        """Called for intermediate results (partial recognition)."""
        logger.debug("Recognizing: %s", evt.result.text[:60] if evt.result.text else "")

    def _on_canceled(self, evt: speechsdk.SpeechRecognitionCanceledEventArgs) -> None:
        """Called when recognition is canceled (usually an error)."""
        self._errors += 1
        logger.error(
            "Recognition canceled — reason=%s, error=%s",
            evt.cancellation_details.reason,
            evt.cancellation_details.error_details,
        )

    def _on_session_stopped(self, evt: speechsdk.SessionEventArgs) -> None:
        """Called when the recognition session stops."""
        logger.info("Speech recognition session stopped")

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
    import asyncio

    async def _on_segment(segment: TranscriptSegment) -> None:
        print(f"  → [{segment.language}] {segment.text}")

    async def _test() -> None:
        # This test only verifies initialisation; it won't connect without keys
        client = AzureSpeechClient(on_segment=_on_segment)
        assert not client.is_running
        assert client.segments_recognised == 0
        print("✓ AzureSpeechClient initialisation test passed")
        print("  (Actual recognition requires valid Azure Speech credentials)")

    asyncio.run(_test())
