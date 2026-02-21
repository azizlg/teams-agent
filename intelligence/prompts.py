"""
prompts.py — Prompt templates for LLM meeting analysis.

Defines structured prompt templates used by the MeetingAnalyzer to extract
insights, summaries, action items, sentiment, and key topics from meeting
transcripts. Supports multilingual input and output.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AnalysisType(str, Enum):
    """Types of analysis the LLM can perform."""

    FULL_REPORT = "full_report"
    EXECUTIVE_SUMMARY = "executive_summary"
    ACTION_ITEMS = "action_items"
    KEY_DECISIONS = "key_decisions"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    TOPIC_EXTRACTION = "topic_extraction"
    PARTICIPANT_ANALYSIS = "participant_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    FOLLOW_UP = "follow_up"


class OutputLanguage(str, Enum):
    """Supported output languages for reports."""

    ENGLISH = "en"
    FRENCH = "fr"
    ARABIC = "ar"
    SPANISH = "es"
    GERMAN = "de"
    PORTUGUESE = "pt"
    ITALIAN = "it"
    CHINESE = "zh"
    JAPANESE = "ja"
    KOREAN = "ko"


# ---------------------------------------------------------------------------
# Prompt template models
# ---------------------------------------------------------------------------

class PromptContext(BaseModel):
    """Context data injected into prompt templates."""

    meeting_title: str = "Untitled Meeting"
    meeting_date: str = ""
    meeting_duration_minutes: float = 0.0
    participant_names: list[str] = Field(default_factory=list)
    languages_detected: list[str] = Field(default_factory=list)
    transcript_text: str = ""
    speaker_summary: list[dict[str, Any]] = Field(default_factory=list)
    output_language: OutputLanguage = OutputLanguage.ENGLISH
    additional_instructions: str = ""


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_BASE = """You are an expert multilingual meeting analyst AI. You analyze \
meeting transcripts and produce structured, actionable reports.

IMPORTANT RULES:
- Be precise and factual — only report what was actually said.
- Distinguish clearly between decisions, action items, and discussion topics.
- Attribute statements to specific speakers when identified.
- If the transcript contains multiple languages, you understand all of them.
- Produce your output in {output_language}.
- Use professional, clear language appropriate for corporate reports.
- Format your output as structured JSON matching the requested schema."""


# ---------------------------------------------------------------------------
# Analysis prompt templates
# ---------------------------------------------------------------------------

PROMPTS: dict[AnalysisType, str] = {
    AnalysisType.FULL_REPORT: """Analyze the following meeting transcript and produce a comprehensive report.

MEETING CONTEXT:
- Title: {meeting_title}
- Date: {meeting_date}
- Duration: {meeting_duration_minutes:.0f} minutes
- Participants: {participants_list}
- Languages detected: {languages_list}

TRANSCRIPT:
{transcript_text}

Produce a JSON response with this exact structure:
{{
    "executive_summary": "2-3 paragraph summary of the meeting",
    "key_topics": [
        {{"topic": "topic name", "summary": "brief summary", "discussed_by": ["speaker names"]}}
    ],
    "decisions_made": [
        {{"decision": "what was decided", "made_by": "who decided", "context": "why"}}
    ],
    "action_items": [
        {{"action": "what needs to be done", "assignee": "who", "deadline": "when if mentioned", "priority": "high/medium/low"}}
    ],
    "key_quotes": [
        {{"quote": "exact quote", "speaker": "who said it", "context": "why it matters"}}
    ],
    "sentiment_overview": {{
        "overall": "positive/neutral/negative/mixed",
        "details": "explanation of the overall tone and dynamics"
    }},
    "risks_and_concerns": [
        {{"risk": "description", "raised_by": "who", "severity": "high/medium/low"}}
    ],
    "follow_up_items": [
        {{"item": "what to follow up on", "responsible": "who", "timeline": "when"}}
    ],
    "participant_engagement": [
        {{"name": "participant", "speaking_time_pct": 0.0, "key_contributions": ["list"]}}
    ]
}}

{additional_instructions}""",

    AnalysisType.EXECUTIVE_SUMMARY: """Analyze the following meeting transcript and produce a concise executive summary.

MEETING CONTEXT:
- Title: {meeting_title}
- Date: {meeting_date}
- Duration: {meeting_duration_minutes:.0f} minutes
- Participants: {participants_list}

TRANSCRIPT:
{transcript_text}

Produce a JSON response:
{{
    "summary": "Comprehensive 3-5 paragraph executive summary covering objectives, key discussions, decisions, and outcomes",
    "headline": "One-line meeting headline",
    "key_takeaways": ["list of 3-5 key takeaways"]
}}

{additional_instructions}""",

    AnalysisType.ACTION_ITEMS: """Extract all action items from this meeting transcript.

TRANSCRIPT:
{transcript_text}

Produce a JSON response:
{{
    "action_items": [
        {{
            "action": "Clear description of what needs to be done",
            "assignee": "Person responsible (or 'Unassigned')",
            "deadline": "Mentioned deadline or 'Not specified'",
            "priority": "high/medium/low",
            "context": "Brief context from the discussion",
            "mentioned_by": "Who raised this item"
        }}
    ],
    "total_count": 0
}}

{additional_instructions}""",

    AnalysisType.KEY_DECISIONS: """Extract all decisions made during this meeting.

TRANSCRIPT:
{transcript_text}

Produce a JSON response:
{{
    "decisions": [
        {{
            "decision": "What was decided",
            "rationale": "Why this decision was made",
            "decided_by": "Who made or proposed the decision",
            "supported_by": ["Others who agreed"],
            "impact": "Expected impact"
        }}
    ],
    "pending_decisions": [
        {{
            "topic": "Undecided topic",
            "options_discussed": ["options"],
            "blocker": "What's preventing a decision"
        }}
    ]
}}

{additional_instructions}""",

    AnalysisType.SENTIMENT_ANALYSIS: """Perform sentiment analysis on this meeting transcript.

MEETING CONTEXT:
- Participants: {participants_list}

TRANSCRIPT:
{transcript_text}

Produce a JSON response:
{{
    "overall_sentiment": "positive/neutral/negative/mixed",
    "confidence": 0.0,
    "tone_description": "Description of the overall meeting tone",
    "participant_sentiments": [
        {{
            "name": "participant",
            "sentiment": "positive/neutral/negative",
            "emotional_markers": ["e.g. enthusiastic, concerned, frustrated"],
            "notable_moments": ["description of notable emotional moments"]
        }}
    ],
    "sentiment_shifts": [
        {{
            "from": "initial sentiment",
            "to": "shifted sentiment",
            "trigger": "what caused the shift",
            "approximate_time": "when in the meeting"
        }}
    ],
    "collaboration_score": 0.0,
    "conflict_indicators": ["any detected conflicts or tensions"]
}}

{additional_instructions}""",

    AnalysisType.TOPIC_EXTRACTION: """Extract and categorize all topics discussed in this meeting.

TRANSCRIPT:
{transcript_text}

Produce a JSON response:
{{
    "topics": [
        {{
            "name": "Topic name",
            "category": "category (e.g. Technical, Business, Planning, Administrative)",
            "time_spent_estimate": "estimated percentage of meeting time",
            "summary": "Brief summary of what was discussed",
            "participants_involved": ["who participated in this topic"],
            "outcome": "resolved/deferred/ongoing",
            "subtopics": ["related subtopics"]
        }}
    ],
    "topic_flow": "Narrative of how topics transitioned during the meeting"
}}

{additional_instructions}""",

    AnalysisType.PARTICIPANT_ANALYSIS: """Analyze each participant's contributions in this meeting.

MEETING CONTEXT:
- Participants: {participants_list}

SPEAKER STATISTICS:
{speaker_summary}

TRANSCRIPT:
{transcript_text}

Produce a JSON response:
{{
    "participants": [
        {{
            "name": "Participant name",
            "role_observed": "Observed role (e.g. Facilitator, Contributor, Observer)",
            "speaking_time_seconds": 0.0,
            "contribution_summary": "Summary of their contributions",
            "key_points_raised": ["list of main points they raised"],
            "questions_asked": ["questions they asked"],
            "agreements": ["things they agreed to"],
            "concerns_raised": ["concerns they expressed"]
        }}
    ],
    "interaction_patterns": "Description of who interacted with whom",
    "leadership_dynamics": "Who led discussions and how"
}}

{additional_instructions}""",

    AnalysisType.RISK_ASSESSMENT: """Identify risks, concerns, and potential issues discussed in this meeting.

TRANSCRIPT:
{transcript_text}

Produce a JSON response:
{{
    "risks": [
        {{
            "risk": "Description of the risk",
            "category": "Technical/Business/Timeline/Resource/Quality",
            "severity": "critical/high/medium/low",
            "likelihood": "high/medium/low",
            "raised_by": "Who raised it",
            "mitigation_discussed": "Any mitigation mentioned",
            "status": "acknowledged/being-addressed/unaddressed"
        }}
    ],
    "overall_risk_level": "low/moderate/elevated/high",
    "recommendations": ["Risk mitigation recommendations based on discussion"]
}}

{additional_instructions}""",

    AnalysisType.FOLLOW_UP: """Identify all items that require follow-up after this meeting.

MEETING CONTEXT:
- Title: {meeting_title}
- Participants: {participants_list}

TRANSCRIPT:
{transcript_text}

Produce a JSON response:
{{
    "follow_ups": [
        {{
            "item": "What needs to follow up",
            "type": "action/decision/information/meeting",
            "responsible": "Who is responsible",
            "timeline": "Expected timeline",
            "dependencies": ["Any dependencies"],
            "priority": "high/medium/low"
        }}
    ],
    "next_meeting_suggested": true,
    "next_meeting_topics": ["Topics for the next meeting"],
    "unresolved_items": ["Items that were not resolved"]
}}

{additional_instructions}""",
}


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(
    analysis_type: AnalysisType,
    context: PromptContext,
) -> tuple[str, str]:
    """
    Build a system prompt and user prompt for the given analysis type.

    Args:
        analysis_type: The type of analysis to perform.
        context: Context data to inject into the template.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    # Build system prompt
    language_names = {
        OutputLanguage.ENGLISH: "English",
        OutputLanguage.FRENCH: "French",
        OutputLanguage.ARABIC: "Arabic",
        OutputLanguage.SPANISH: "Spanish",
        OutputLanguage.GERMAN: "German",
        OutputLanguage.PORTUGUESE: "Portuguese",
        OutputLanguage.ITALIAN: "Italian",
        OutputLanguage.CHINESE: "Chinese",
        OutputLanguage.JAPANESE: "Japanese",
        OutputLanguage.KOREAN: "Korean",
    }
    output_lang_name = language_names.get(context.output_language, "English")
    system_prompt = SYSTEM_PROMPT_BASE.format(output_language=output_lang_name)

    # Build user prompt from template
    template = PROMPTS[analysis_type]
    participants_list = ", ".join(context.participant_names) or "Not identified"
    languages_list = ", ".join(context.languages_detected) or "Not detected"

    speaker_summary_text = ""
    for s in context.speaker_summary:
        name = s.get("participant_name", s.get("speaker_label", "Unknown"))
        time_s = s.get("total_speaking_time", 0)
        speaker_summary_text += f"- {name}: {time_s:.0f}s speaking time\n"

    user_prompt = template.format(
        meeting_title=context.meeting_title,
        meeting_date=context.meeting_date,
        meeting_duration_minutes=context.meeting_duration_minutes,
        participants_list=participants_list,
        languages_list=languages_list,
        transcript_text=context.transcript_text,
        speaker_summary=speaker_summary_text or "No speaker statistics available",
        additional_instructions=context.additional_instructions,
    )

    return system_prompt, user_prompt


# ---------------------------------------------------------------------------
# Unit test stub
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ctx = PromptContext(
        meeting_title="Q4 Planning",
        meeting_date="2024-12-01",
        meeting_duration_minutes=45,
        participant_names=["Alice", "Bob", "Charlie"],
        languages_detected=["en-US", "fr-FR"],
        transcript_text="Alice: Let's discuss the roadmap...\nBob: I think we should...",
    )

    for atype in AnalysisType:
        sys_p, user_p = build_prompt(atype, ctx)
        assert len(sys_p) > 0
        assert len(user_p) > 0
        assert "Q4 Planning" in user_p or atype in (
            AnalysisType.ACTION_ITEMS,
            AnalysisType.KEY_DECISIONS,
            AnalysisType.TOPIC_EXTRACTION,
            AnalysisType.RISK_ASSESSMENT,
        )

    print(f"✓ All {len(AnalysisType)} prompt templates build successfully")
