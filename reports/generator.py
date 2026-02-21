"""
generator.py — Report generation engine.

Takes analysis results from the MeetingAnalyzer and renders them
into structured reports using templates. Supports Markdown, HTML,
and structured data output. Delegates to exporters for DOCX/PDF.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from intelligence.analyzer import AnalysisResult, FullMeetingReport
from reports.templates import (
    ReportFormat,
    ReportTemplate,
    SectionConfig,
    get_template,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Report data model
# ---------------------------------------------------------------------------

class GeneratedReport(BaseModel):
    """A fully generated report ready for export."""

    meeting_id: str
    meeting_title: str = ""
    template_name: str = ""
    format: ReportFormat = ReportFormat.MARKDOWN
    content: str = ""  # rendered content (MD, HTML, or JSON)
    sections: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    generated_at: str = ""
    generation_time_seconds: float = 0.0


# ---------------------------------------------------------------------------
# ReportGenerator
# ---------------------------------------------------------------------------

class ReportGenerator:
    """
    Generates formatted reports from analysis results.

    Takes a FullMeetingReport (or individual AnalysisResults) and
    renders them using a template into the requested output format.

    Usage:
        gen = ReportGenerator()
        report = gen.generate(
            analysis_report=full_report,
            template_name="full_report",
            output_format=ReportFormat.MARKDOWN,
        )
    """

    def __init__(self) -> None:
        logger.info("ReportGenerator initialised")

    # ------------------------------------------------------------------
    # Main generation
    # ------------------------------------------------------------------

    def generate(
        self,
        analysis_report: FullMeetingReport,
        *,
        template_name: str = "full_report",
        output_format: ReportFormat = ReportFormat.MARKDOWN,
        meeting_date: str = "",
    ) -> GeneratedReport:
        """
        Generate a report from analysis results.

        Args:
            analysis_report: Full meeting analysis report.
            template_name: Template to use.
            output_format: Desired output format.
            meeting_date: Meeting date string.

        Returns:
            GeneratedReport with rendered content.
        """
        start_time = time.time()
        template = get_template(template_name)

        logger.info(
            "Generating %s report for meeting %s using template '%s'",
            output_format.value,
            analysis_report.meeting_id,
            template_name,
        )

        # Collect section data
        sections = self._collect_sections(template, analysis_report)

        # Render based on format
        if output_format == ReportFormat.MARKDOWN:
            content = self._render_markdown(template, sections, analysis_report, meeting_date)
        elif output_format == ReportFormat.HTML:
            content = self._render_html(template, sections, analysis_report, meeting_date)
        elif output_format == ReportFormat.JSON:
            import json
            content = json.dumps({
                "meeting_id": analysis_report.meeting_id,
                "meeting_title": analysis_report.meeting_title,
                "sections": sections,
            }, indent=2, default=str)
        else:
            content = self._render_markdown(template, sections, analysis_report, meeting_date)

        now = datetime.now(timezone.utc).isoformat()
        result = GeneratedReport(
            meeting_id=analysis_report.meeting_id,
            meeting_title=analysis_report.meeting_title,
            template_name=template_name,
            format=output_format,
            content=content,
            sections=sections,
            metadata={
                "total_tokens": analysis_report.total_tokens_used,
                "analysis_count": len(analysis_report.analyses),
                "template": template_name,
            },
            generated_at=now,
            generation_time_seconds=time.time() - start_time,
        )

        logger.info(
            "Report generated — %d chars, %.2f s",
            len(content),
            result.generation_time_seconds,
        )
        return result

    # ------------------------------------------------------------------
    # Section collection
    # ------------------------------------------------------------------

    def _collect_sections(
        self,
        template: ReportTemplate,
        report: FullMeetingReport,
    ) -> list[dict[str, Any]]:
        """Gather data for each template section from analysis results."""
        sections: list[dict[str, Any]] = []

        for section_config in sorted(template.sections, key=lambda s: s.order):
            data = self._extract_section_data(section_config.key, report)
            sections.append({
                "key": section_config.key,
                "title": section_config.title,
                "icon": section_config.icon,
                "data": data,
                "has_data": bool(data),
            })

        return sections

    def _extract_section_data(
        self,
        key: str,
        report: FullMeetingReport,
    ) -> Any:
        """Extract data for a section from the analysis results."""
        # Map section keys to analysis type results
        key_to_analysis = {
            "executive_summary": "executive_summary",
            "key_topics": "topic_extraction",
            "decisions_made": "key_decisions",
            "action_items": "action_items",
            "sentiment_analysis": "sentiment_analysis",
            "risks_and_concerns": "risk_assessment",
            "participant_engagement": "participant_analysis",
            "follow_up_items": "follow_up",
            "sentiment_overview": "sentiment_analysis",
        }

        analysis_key = key_to_analysis.get(key, key)
        analysis_result = report.analyses.get(analysis_key)

        if not analysis_result or not analysis_result.success:
            return None

        data = analysis_result.data

        # Try to extract the specific sub-key
        if key in data:
            return data[key]

        # For some analyses, the entire data dict is the section
        return data

    # ------------------------------------------------------------------
    # Markdown renderer
    # ------------------------------------------------------------------

    def _render_markdown(
        self,
        template: ReportTemplate,
        sections: list[dict[str, Any]],
        report: FullMeetingReport,
        meeting_date: str,
    ) -> str:
        """Render the report as Markdown."""
        lines: list[str] = []

        # Header
        lines.append(f"# {template.header_title}")
        subtitle = template.subtitle_template.format(
            meeting_title=report.meeting_title or "Untitled Meeting",
            meeting_date=meeting_date or "Date not specified",
        )
        lines.append(f"*{subtitle}*")
        lines.append("")

        # Table of contents
        if template.include_toc and len(sections) > 2:
            lines.append("## Table of Contents")
            for sec in sections:
                if sec["has_data"]:
                    anchor = sec["title"].lower().replace(" ", "-").replace("&", "and")
                    lines.append(f"- [{sec['icon']} {sec['title']}](#{anchor})")
            lines.append("")

        # Sections
        for sec in sections:
            if not sec["has_data"]:
                continue

            lines.append(f"## {sec['icon']} {sec['title']}")
            lines.append("")
            lines.extend(self._render_section_data_md(sec["data"]))
            lines.append("")

        # Footer
        lines.append("---")
        lines.append(f"*{template.footer_text}*  ")
        lines.append(f"*Generated: {report.generated_at}*")

        return "\n".join(lines)

    def _render_section_data_md(self, data: Any) -> list[str]:
        """Render section data as Markdown lines."""
        lines: list[str] = []

        if isinstance(data, str):
            lines.append(data)
            lines.append("")

        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    lines.extend(self._render_dict_as_md(item))
                else:
                    lines.append(f"- {item}")
            lines.append("")

        elif isinstance(data, dict):
            for key, value in data.items():
                if key.startswith("_"):
                    continue

                display_key = key.replace("_", " ").title()

                if isinstance(value, str):
                    lines.append(f"**{display_key}:** {value}")
                    lines.append("")
                elif isinstance(value, list):
                    lines.append(f"### {display_key}")
                    for item in value:
                        if isinstance(item, dict):
                            lines.extend(self._render_dict_as_md(item))
                        else:
                            lines.append(f"- {item}")
                    lines.append("")
                elif isinstance(value, dict):
                    lines.append(f"### {display_key}")
                    for k, v in value.items():
                        lines.append(f"- **{k.replace('_', ' ').title()}:** {v}")
                    lines.append("")
                else:
                    lines.append(f"**{display_key}:** {value}")

        return lines

    def _render_dict_as_md(self, item: dict[str, Any]) -> list[str]:
        """Render a single dict item as a Markdown list entry."""
        lines: list[str] = []
        first_key = next(iter(item), None)
        if first_key:
            first_val = item[first_key]
            display_key = first_key.replace("_", " ").title()
            lines.append(f"- **{display_key}:** {first_val}")
            for k, v in item.items():
                if k == first_key or k.startswith("_"):
                    continue
                display_k = k.replace("_", " ").title()
                if isinstance(v, list):
                    lines.append(f"  - {display_k}: {', '.join(str(x) for x in v)}")
                else:
                    lines.append(f"  - {display_k}: {v}")
        return lines

    # ------------------------------------------------------------------
    # HTML renderer
    # ------------------------------------------------------------------

    def _render_html(
        self,
        template: ReportTemplate,
        sections: list[dict[str, Any]],
        report: FullMeetingReport,
        meeting_date: str,
    ) -> str:
        """Render the report as styled HTML."""
        subtitle = template.subtitle_template.format(
            meeting_title=report.meeting_title or "Untitled Meeting",
            meeting_date=meeting_date or "Date not specified",
        )

        html_parts: list[str] = [
            "<!DOCTYPE html>",
            "<html lang='en'>",
            "<head>",
            f"  <meta charset='UTF-8'>",
            f"  <title>{template.header_title}</title>",
            f"  <style>",
            f"    :root {{ --primary: {template.primary_color}; --accent: {template.accent_color}; }}",
            f"    body {{ font-family: {template.font_family}; line-height: 1.6; max-width: 900px; margin: 0 auto; padding: 2rem; color: #333; }}",
            f"    h1 {{ color: var(--primary); border-bottom: 3px solid var(--primary); padding-bottom: 0.5rem; }}",
            f"    h2 {{ color: var(--accent); margin-top: 2rem; }}",
            f"    .subtitle {{ color: #666; font-size: 1.1rem; margin-bottom: 2rem; }}",
            f"    .section {{ margin-bottom: 2rem; }}",
            f"    .item {{ background: #f8f9fa; border-left: 4px solid var(--primary); padding: 1rem; margin: 0.5rem 0; border-radius: 0 8px 8px 0; }}",
            f"    .item strong {{ color: var(--accent); }}",
            f"    .priority-high {{ border-left-color: #dc3545; }}",
            f"    .priority-medium {{ border-left-color: #ffc107; }}",
            f"    .priority-low {{ border-left-color: #28a745; }}",
            f"    table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}",
            f"    th, td {{ padding: 0.75rem; text-align: left; border-bottom: 1px solid #dee2e6; }}",
            f"    th {{ background: var(--primary); color: white; }}",
            f"    .footer {{ margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #dee2e6; color: #999; font-size: 0.85rem; }}",
            f"  </style>",
            "</head>",
            "<body>",
            f"  <h1>{template.header_title}</h1>",
            f"  <p class='subtitle'>{subtitle}</p>",
        ]

        for sec in sections:
            if not sec["has_data"]:
                continue
            html_parts.append(f"  <div class='section'>")
            html_parts.append(f"    <h2>{sec['icon']} {sec['title']}</h2>")
            html_parts.extend(self._render_section_data_html(sec["data"]))
            html_parts.append(f"  </div>")

        html_parts.extend([
            f"  <div class='footer'>",
            f"    <p>{template.footer_text} | Generated: {report.generated_at}</p>",
            f"  </div>",
            "</body>",
            "</html>",
        ])

        return "\n".join(html_parts)

    def _render_section_data_html(self, data: Any) -> list[str]:
        """Render section data as HTML."""
        parts: list[str] = []

        if isinstance(data, str):
            paragraphs = data.split("\n\n")
            for p in paragraphs:
                parts.append(f"    <p>{p}</p>")

        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    priority = str(item.get("priority", "")).lower()
                    css_class = f"item priority-{priority}" if priority else "item"
                    parts.append(f"    <div class='{css_class}'>")
                    for k, v in item.items():
                        if k.startswith("_"):
                            continue
                        display_k = k.replace("_", " ").title()
                        if isinstance(v, list):
                            v_str = ", ".join(str(x) for x in v)
                            parts.append(f"      <p><strong>{display_k}:</strong> {v_str}</p>")
                        else:
                            parts.append(f"      <p><strong>{display_k}:</strong> {v}</p>")
                    parts.append(f"    </div>")
                else:
                    parts.append(f"    <p>• {item}</p>")

        elif isinstance(data, dict):
            for k, v in data.items():
                if k.startswith("_"):
                    continue
                display_k = k.replace("_", " ").title()
                if isinstance(v, str):
                    parts.append(f"    <p><strong>{display_k}:</strong> {v}</p>")
                elif isinstance(v, list):
                    parts.append(f"    <h3>{display_k}</h3>")
                    parts.extend(self._render_section_data_html(v))
                else:
                    parts.append(f"    <p><strong>{display_k}:</strong> {v}</p>")

        return parts


# ---------------------------------------------------------------------------
# Unit test stub
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    gen = ReportGenerator()

    # Create mock analysis report
    mock_report = FullMeetingReport(
        meeting_id="test-mtg",
        meeting_title="Q4 Planning Session",
        generated_at="2024-12-01T10:00:00Z",
        analyses={
            "executive_summary": AnalysisResult(
                analysis_type="executive_summary",
                meeting_id="test-mtg",
                data={
                    "summary": "The team discussed Q4 priorities and allocated resources.",
                    "headline": "Q4 Priorities Defined",
                    "key_takeaways": ["Budget approved", "Timeline set", "Risks identified"],
                },
            ),
            "action_items": AnalysisResult(
                analysis_type="action_items",
                meeting_id="test-mtg",
                data={
                    "action_items": [
                        {"action": "Prepare budget draft", "assignee": "Alice", "deadline": "Dec 15", "priority": "high"},
                        {"action": "Review vendor contracts", "assignee": "Bob", "deadline": "Dec 20", "priority": "medium"},
                    ],
                    "total_count": 2,
                },
            ),
        },
    )

    # Test Markdown
    md_report = gen.generate(mock_report, output_format=ReportFormat.MARKDOWN, meeting_date="December 1, 2024")
    assert "Q4 Planning Session" in md_report.content
    assert "Action Items" in md_report.content

    # Test HTML
    html_report = gen.generate(mock_report, output_format=ReportFormat.HTML, meeting_date="December 1, 2024")
    assert "<html" in html_report.content

    # Test JSON
    json_report = gen.generate(mock_report, output_format=ReportFormat.JSON)
    assert "test-mtg" in json_report.content

    print(f"✓ ReportGenerator tests passed")
    print(f"  MD: {len(md_report.content)} chars")
    print(f"  HTML: {len(html_report.content)} chars")
    print(f"  JSON: {len(json_report.content)} chars")
