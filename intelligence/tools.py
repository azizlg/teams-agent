"""
tools.py — LangChain tool definitions for meeting analysis.

Defines tools that the LLM agent can use during analysis, such as
querying transcript data, looking up participant information, and
searching for specific discussion topics.
"""

from __future__ import annotations

import json
import logging
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from storage.database import Database

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool input/output models
# ---------------------------------------------------------------------------

class SearchTranscriptInput(BaseModel):
    """Input for searching within a meeting transcript."""

    query: str = Field(..., description="Search query — keyword or phrase to find in the transcript")
    meeting_id: str = Field(..., description="Meeting ID to search within")
    language: str | None = Field(None, description="Filter by language code (e.g. 'en-US')")
    speaker: str | None = Field(None, description="Filter by speaker name or ID")


class GetSpeakerStatsInput(BaseModel):
    """Input for retrieving speaker statistics."""

    meeting_id: str = Field(..., description="Meeting ID")
    speaker_name: str | None = Field(None, description="Specific speaker name, or omit for all")


class GetMeetingInfoInput(BaseModel):
    """Input for retrieving meeting metadata."""

    meeting_id: str = Field(..., description="Meeting ID")


class GetTopicSegmentsInput(BaseModel):
    """Input for finding transcript segments about a specific topic."""

    meeting_id: str = Field(..., description="Meeting ID")
    topic: str = Field(..., description="Topic keyword or phrase")
    context_window: int = Field(3, description="Number of surrounding segments to include")


class GetTimeRangeInput(BaseModel):
    """Input for getting transcript within a time range."""

    meeting_id: str = Field(..., description="Meeting ID")
    start_minutes: float = Field(..., description="Start time in minutes from meeting beginning")
    end_minutes: float = Field(..., description="End time in minutes from meeting beginning")


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

class MeetingTools:
    """
    Collection of tools available to the LLM during meeting analysis.

    Each tool provides structured access to meeting data, enabling
    the LLM to query specific information rather than processing
    the entire transcript at once.
    """

    def __init__(self, db: Database | None = None) -> None:
        self._db = db
        # In-memory transcript cache for the current meeting
        self._transcript_cache: dict[str, list[dict[str, Any]]] = {}
        self._meeting_cache: dict[str, dict[str, Any]] = {}

    def set_meeting_data(
        self,
        meeting_id: str,
        transcript_segments: list[dict[str, Any]],
        meeting_info: dict[str, Any] | None = None,
    ) -> None:
        """
        Preload meeting data for tool access.

        Call this before running analysis so tools can access the data
        without requiring database queries.
        """
        self._transcript_cache[meeting_id] = transcript_segments
        if meeting_info:
            self._meeting_cache[meeting_id] = meeting_info

    # ------------------------------------------------------------------
    # Tool: Search transcript
    # ------------------------------------------------------------------

    async def search_transcript(self, input_data: SearchTranscriptInput) -> str:
        """
        Search the meeting transcript for specific keywords or phrases.

        Returns matching segments with context.
        """
        segments = self._transcript_cache.get(input_data.meeting_id, [])
        if not segments:
            return json.dumps({"error": "Meeting not found", "results": []})

        query_lower = input_data.query.lower()
        results = []

        for i, seg in enumerate(segments):
            text = seg.get("text", "")
            if query_lower not in text.lower():
                continue

            # Apply filters
            if input_data.language and seg.get("language") != input_data.language:
                continue
            if input_data.speaker and seg.get("speaker_id") != input_data.speaker:
                continue

            results.append({
                "index": i,
                "text": text,
                "speaker": seg.get("speaker_id", "unknown"),
                "language": seg.get("language", "unknown"),
                "timestamp": seg.get("timestamp", 0),
                "start": seg.get("start_time", 0),
                "end": seg.get("end_time", 0),
            })

        return json.dumps({
            "query": input_data.query,
            "total_matches": len(results),
            "results": results[:20],  # cap at 20
        })

    # ------------------------------------------------------------------
    # Tool: Get speaker statistics
    # ------------------------------------------------------------------

    async def get_speaker_stats(self, input_data: GetSpeakerStatsInput) -> str:
        """Get speaking time and contribution statistics for participants."""
        segments = self._transcript_cache.get(input_data.meeting_id, [])
        if not segments:
            return json.dumps({"error": "Meeting not found"})

        # Aggregate by speaker
        stats: dict[str, dict[str, Any]] = {}
        for seg in segments:
            speaker = seg.get("speaker_id") or "unknown"
            if input_data.speaker_name and speaker != input_data.speaker_name:
                continue

            if speaker not in stats:
                stats[speaker] = {
                    "speaker": speaker,
                    "segment_count": 0,
                    "total_duration": 0.0,
                    "languages_used": set(),
                    "word_count": 0,
                }

            stats[speaker]["segment_count"] += 1
            duration = seg.get("end_time", 0) - seg.get("start_time", 0)
            stats[speaker]["total_duration"] += max(0, duration)
            stats[speaker]["languages_used"].add(seg.get("language", "unknown"))
            stats[speaker]["word_count"] += len(seg.get("text", "").split())

        # Convert sets to lists for JSON
        result = []
        for s in stats.values():
            s["languages_used"] = list(s["languages_used"])
            result.append(s)

        result.sort(key=lambda x: x["total_duration"], reverse=True)

        return json.dumps({"speakers": result})

    # ------------------------------------------------------------------
    # Tool: Get meeting info
    # ------------------------------------------------------------------

    async def get_meeting_info(self, input_data: GetMeetingInfoInput) -> str:
        """Get meeting metadata including title, duration, and participants."""
        info = self._meeting_cache.get(input_data.meeting_id)
        if not info:
            segments = self._transcript_cache.get(input_data.meeting_id, [])
            if not segments:
                return json.dumps({"error": "Meeting not found"})

            # Build info from transcript
            speakers = set()
            languages = set()
            for seg in segments:
                if seg.get("speaker_id"):
                    speakers.add(seg["speaker_id"])
                if seg.get("language"):
                    languages.add(seg["language"])

            info = {
                "meeting_id": input_data.meeting_id,
                "total_segments": len(segments),
                "speakers": list(speakers),
                "languages": list(languages),
            }

        return json.dumps(info)

    # ------------------------------------------------------------------
    # Tool: Get topic segments
    # ------------------------------------------------------------------

    async def get_topic_segments(self, input_data: GetTopicSegmentsInput) -> str:
        """Find transcript segments related to a specific topic with surrounding context."""
        segments = self._transcript_cache.get(input_data.meeting_id, [])
        if not segments:
            return json.dumps({"error": "Meeting not found"})

        topic_lower = input_data.topic.lower()
        match_indices: set[int] = set()

        # Find matching segments
        for i, seg in enumerate(segments):
            if topic_lower in seg.get("text", "").lower():
                # Add context window
                for j in range(
                    max(0, i - input_data.context_window),
                    min(len(segments), i + input_data.context_window + 1),
                ):
                    match_indices.add(j)

        # Build result with context
        results = []
        for i in sorted(match_indices):
            seg = segments[i]
            results.append({
                "index": i,
                "text": seg.get("text", ""),
                "speaker": seg.get("speaker_id", "unknown"),
                "is_match": topic_lower in seg.get("text", "").lower(),
                "start": seg.get("start_time", 0),
                "end": seg.get("end_time", 0),
            })

        return json.dumps({
            "topic": input_data.topic,
            "segments_found": len(results),
            "results": results,
        })

    # ------------------------------------------------------------------
    # Tool: Get time range
    # ------------------------------------------------------------------

    async def get_time_range(self, input_data: GetTimeRangeInput) -> str:
        """Get transcript segments within a specific time range."""
        segments = self._transcript_cache.get(input_data.meeting_id, [])
        if not segments:
            return json.dumps({"error": "Meeting not found"})

        start_sec = input_data.start_minutes * 60
        end_sec = input_data.end_minutes * 60

        results = []
        for seg in segments:
            seg_start = seg.get("start_time", seg.get("timestamp", 0))
            if start_sec <= seg_start <= end_sec:
                results.append({
                    "text": seg.get("text", ""),
                    "speaker": seg.get("speaker_id", "unknown"),
                    "language": seg.get("language", "unknown"),
                    "start": seg_start,
                })

        return json.dumps({
            "time_range": f"{input_data.start_minutes:.0f}-{input_data.end_minutes:.0f} min",
            "segments": results,
        })

    # ------------------------------------------------------------------
    # Get tool definitions for LangChain
    # ------------------------------------------------------------------

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """
        Return tool definitions in a format suitable for LangChain.

        Each definition includes name, description, and input schema.
        """
        return [
            {
                "name": "search_transcript",
                "description": "Search the meeting transcript for specific keywords or phrases. Returns matching segments with speaker and timing info.",
                "input_schema": SearchTranscriptInput.model_json_schema(),
                "func": self.search_transcript,
            },
            {
                "name": "get_speaker_stats",
                "description": "Get speaking time and contribution statistics for meeting participants.",
                "input_schema": GetSpeakerStatsInput.model_json_schema(),
                "func": self.get_speaker_stats,
            },
            {
                "name": "get_meeting_info",
                "description": "Get meeting metadata including title, duration, participants, and languages.",
                "input_schema": GetMeetingInfoInput.model_json_schema(),
                "func": self.get_meeting_info,
            },
            {
                "name": "get_topic_segments",
                "description": "Find transcript segments related to a specific topic with surrounding context for better understanding.",
                "input_schema": GetTopicSegmentsInput.model_json_schema(),
                "func": self.get_topic_segments,
            },
            {
                "name": "get_time_range",
                "description": "Get all transcript segments within a specific time range (in minutes from meeting start).",
                "input_schema": GetTimeRangeInput.model_json_schema(),
                "func": self.get_time_range,
            },
        ]


# ---------------------------------------------------------------------------
# Unit test stub
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    async def _test() -> None:
        tools = MeetingTools()

        # Preload test data
        tools.set_meeting_data(
            meeting_id="test-mtg",
            transcript_segments=[
                {"text": "Let's discuss the budget", "speaker_id": "Alice", "language": "en-US", "start_time": 0, "end_time": 5},
                {"text": "I think we need more resources", "speaker_id": "Bob", "language": "en-US", "start_time": 5, "end_time": 10},
                {"text": "The timeline is tight", "speaker_id": "Alice", "language": "en-US", "start_time": 10, "end_time": 15},
            ],
        )

        # Test search
        result = await tools.search_transcript(SearchTranscriptInput(query="budget", meeting_id="test-mtg"))
        parsed = json.loads(result)
        assert parsed["total_matches"] == 1

        # Test speaker stats
        result = await tools.get_speaker_stats(GetSpeakerStatsInput(meeting_id="test-mtg"))
        parsed = json.loads(result)
        assert len(parsed["speakers"]) == 2

        # Test tool definitions
        defs = tools.get_tool_definitions()
        assert len(defs) == 5

        print(f"✓ MeetingTools test passed — {len(defs)} tools defined")

    asyncio.run(_test())
