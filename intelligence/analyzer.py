"""
analyzer.py — LLM-powered meeting analysis engine.

Orchestrates Groq API (Llama 3 / Mixtral) to analyze meeting transcripts
and produce structured reports. Supports multiple analysis types,
output in 10 languages, and tool-augmented analysis.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from pydantic import BaseModel, Field

from config.settings import settings
from intelligence.prompts import (
    AnalysisType,
    OutputLanguage,
    PromptContext,
    build_prompt,
)
from intelligence.tools import MeetingTools

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

class AnalysisResult(BaseModel):
    """Result of a single analysis run."""

    analysis_type: AnalysisType
    meeting_id: str
    data: dict[str, Any] = Field(default_factory=dict)
    raw_response: str = ""
    model_used: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    processing_time_seconds: float = 0.0
    success: bool = True
    error: str | None = None


class FullMeetingReport(BaseModel):
    """Aggregated report containing multiple analyses."""

    meeting_id: str
    meeting_title: str = ""
    analyses: dict[str, AnalysisResult] = Field(default_factory=dict)
    generated_at: str = ""
    total_processing_time_seconds: float = 0.0
    total_tokens_used: int = 0


# ---------------------------------------------------------------------------
# MeetingAnalyzer
# ---------------------------------------------------------------------------

class MeetingAnalyzer:
    """
    LLM-powered meeting analysis engine.

    Uses Groq API (Llama 3 / Mixtral) to analyze transcripts and produce
    structured insights. Supports:
    - Multiple analysis types (summary, actions, sentiment, etc.)
    - Tool-augmented analysis for large transcripts
    - Output in 10 languages
    - Token tracking and cost estimation

    Usage:
        analyzer = MeetingAnalyzer()
        result = await analyzer.analyze(
            meeting_id="mtg-123",
            transcript_segments=[...],
            analysis_type=AnalysisType.FULL_REPORT,
        )
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
    ) -> None:
        self._api_key = api_key or settings.groq.api_key
        self._model = model or settings.groq.model
        self._max_tokens = max_tokens
        self._client: Any = None
        self._tools = MeetingTools()

        logger.info("MeetingAnalyzer initialised — model=%s", self._model)

    # ------------------------------------------------------------------
    # Client lifecycle
    # ------------------------------------------------------------------

    async def _ensure_client(self) -> Any:
        """Lazily initialise the Groq client."""
        if self._client is None:
            from groq import AsyncGroq
            self._client = AsyncGroq(api_key=self._api_key)
            logger.info("Groq client initialised")
        return self._client

    # ------------------------------------------------------------------
    # Single analysis
    # ------------------------------------------------------------------

    async def analyze(
        self,
        meeting_id: str,
        transcript_segments: list[dict[str, Any]],
        analysis_type: AnalysisType = AnalysisType.FULL_REPORT,
        *,
        meeting_title: str = "",
        meeting_date: str = "",
        meeting_duration_minutes: float = 0.0,
        participant_names: list[str] | None = None,
        speaker_summary: list[dict[str, Any]] | None = None,
        output_language: OutputLanguage = OutputLanguage.ENGLISH,
        additional_instructions: str = "",
    ) -> AnalysisResult:
        """
        Run a specific type of analysis on a meeting transcript.

        Args:
            meeting_id: Meeting identifier.
            transcript_segments: List of transcript dicts.
            analysis_type: What kind of analysis to perform.
            meeting_title: Optional meeting title.
            meeting_date: Optional date string.
            meeting_duration_minutes: Meeting duration.
            participant_names: List of participant names.
            speaker_summary: Speaker statistics.
            output_language: Language for the output.
            additional_instructions: Extra instructions for the LLM.

        Returns:
            AnalysisResult with structured data.
        """
        start_time = time.time()
        logger.info(
            "Starting %s analysis for meeting %s",
            analysis_type.value,
            meeting_id,
        )

        # Preload tools with meeting data
        self._tools.set_meeting_data(meeting_id, transcript_segments)

        # Build transcript text
        transcript_text = self._format_transcript(transcript_segments)

        # Detect languages
        languages = list(set(
            seg.get("language", "unknown")
            for seg in transcript_segments
            if seg.get("language")
        ))

        # Build prompt
        context = PromptContext(
            meeting_title=meeting_title or "Untitled Meeting",
            meeting_date=meeting_date,
            meeting_duration_minutes=meeting_duration_minutes,
            participant_names=participant_names or [],
            languages_detected=languages,
            transcript_text=transcript_text,
            speaker_summary=speaker_summary or [],
            output_language=output_language,
            additional_instructions=additional_instructions,
        )

        system_prompt, user_prompt = build_prompt(analysis_type, context)

        try:
            # Call Groq API (OpenAI-compatible format)
            client = await self._ensure_client()
            response = await client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
            )

            # Extract response
            raw_text = response.choices[0].message.content
            parsed_data = self._parse_json_response(raw_text)

            # Token usage
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0

            result = AnalysisResult(
                analysis_type=analysis_type,
                meeting_id=meeting_id,
                data=parsed_data,
                raw_response=raw_text,
                model_used=self._model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                processing_time_seconds=time.time() - start_time,
            )

            logger.info(
                "%s analysis complete — %d input + %d output tokens, %.1f s",
                analysis_type.value,
                result.input_tokens,
                result.output_tokens,
                result.processing_time_seconds,
            )

            return result

        except Exception as e:
            logger.exception("Analysis failed for %s", analysis_type.value)
            return AnalysisResult(
                analysis_type=analysis_type,
                meeting_id=meeting_id,
                success=False,
                error=str(e),
                processing_time_seconds=time.time() - start_time,
            )

    # ------------------------------------------------------------------
    # Full meeting report
    # ------------------------------------------------------------------

    async def generate_full_report(
        self,
        meeting_id: str,
        transcript_segments: list[dict[str, Any]],
        *,
        meeting_title: str = "",
        meeting_date: str = "",
        meeting_duration_minutes: float = 0.0,
        participant_names: list[str] | None = None,
        speaker_summary: list[dict[str, Any]] | None = None,
        output_language: OutputLanguage = OutputLanguage.ENGLISH,
        analysis_types: list[AnalysisType] | None = None,
    ) -> FullMeetingReport:
        """
        Generate a comprehensive meeting report with multiple analyses.

        Runs specified analysis types (defaults to all key types)
        and aggregates results into a single report.
        """
        if analysis_types is None:
            analysis_types = [
                AnalysisType.EXECUTIVE_SUMMARY,
                AnalysisType.ACTION_ITEMS,
                AnalysisType.KEY_DECISIONS,
                AnalysisType.SENTIMENT_ANALYSIS,
                AnalysisType.TOPIC_EXTRACTION,
                AnalysisType.RISK_ASSESSMENT,
                AnalysisType.FOLLOW_UP,
            ]

        start_time = time.time()
        logger.info(
            "Generating full report for meeting %s — %d analysis types",
            meeting_id,
            len(analysis_types),
        )

        report = FullMeetingReport(
            meeting_id=meeting_id,
            meeting_title=meeting_title,
        )

        # Run analyses sequentially to stay within rate limits
        for atype in analysis_types:
            result = await self.analyze(
                meeting_id=meeting_id,
                transcript_segments=transcript_segments,
                analysis_type=atype,
                meeting_title=meeting_title,
                meeting_date=meeting_date,
                meeting_duration_minutes=meeting_duration_minutes,
                participant_names=participant_names,
                speaker_summary=speaker_summary,
                output_language=output_language,
            )
            report.analyses[atype.value] = result
            report.total_tokens_used += result.input_tokens + result.output_tokens

        report.total_processing_time_seconds = time.time() - start_time
        from datetime import datetime, timezone
        report.generated_at = datetime.now(timezone.utc).isoformat()

        logger.info(
            "Full report generated — %d analyses, %d tokens, %.1f s",
            len(report.analyses),
            report.total_tokens_used,
            report.total_processing_time_seconds,
        )

        return report

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_transcript(self, segments: list[dict[str, Any]]) -> str:
        """Format transcript segments into readable text."""
        lines: list[str] = []
        for seg in segments:
            speaker = seg.get("speaker_id") or "Unknown"
            text = seg.get("text", "").strip()
            lang = seg.get("language", "")
            timestamp = seg.get("timestamp", 0)

            if not text:
                continue

            # Format timestamp as MM:SS
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)

            lang_tag = f" [{lang}]" if lang and lang != "unknown" else ""
            lines.append(f"[{minutes:02d}:{seconds:02d}] {speaker}{lang_tag}: {text}")

        return "\n".join(lines)

    def _parse_json_response(self, raw_text: str) -> dict[str, Any]:
        """
        Parse the LLM's JSON response, handling common formatting issues.

        The LLM might wrap JSON in markdown code blocks or include
        explanatory text before/after the JSON.
        """
        text = raw_text.strip()

        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            try:
                return json.loads(text[start:end].strip())
            except (json.JSONDecodeError, ValueError):
                pass

        if "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            try:
                return json.loads(text[start:end].strip())
            except (json.JSONDecodeError, ValueError):
                pass

        # Try finding JSON object boundaries
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end != -1:
            try:
                return json.loads(text[brace_start : brace_end + 1])
            except json.JSONDecodeError:
                pass

        # Fallback: return raw text as data
        logger.warning("Could not parse JSON from LLM response — returning raw text")
        return {"raw_text": text, "_parse_error": True}


# ---------------------------------------------------------------------------
# Unit test stub
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    async def _test() -> None:
        analyzer = MeetingAnalyzer()

        # Test transcript formatting
        segments = [
            {"speaker_id": "Alice", "text": "Hello", "language": "en-US", "timestamp": 0},
            {"speaker_id": "Bob", "text": "Bonjour", "language": "fr-FR", "timestamp": 65},
            {"speaker_id": "Alice", "text": "Let's begin", "timestamp": 120},
        ]
        formatted = analyzer._format_transcript(segments)
        assert "[00:00] Alice [en-US]: Hello" in formatted
        assert "[01:05] Bob [fr-FR]: Bonjour" in formatted
        assert "[02:00] Alice: Let's begin" in formatted

        # Test JSON parsing
        assert analyzer._parse_json_response('{"key": "value"}') == {"key": "value"}
        assert analyzer._parse_json_response('```json\n{"key": "value"}\n```') == {"key": "value"}
        assert analyzer._parse_json_response('Some text {"key": "value"} more text') == {"key": "value"}
        assert "_parse_error" in analyzer._parse_json_response("not json at all")

        print("MeetingAnalyzer test passed")
        print(f"  Formatted transcript:\n{formatted}")

    import asyncio
    asyncio.run(_test())
