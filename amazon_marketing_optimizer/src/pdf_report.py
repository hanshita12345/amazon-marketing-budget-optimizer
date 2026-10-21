"""Professional PDF executive management report generator using ReportLab.

Produces Amazon-styled executive reports with KPI summaries, channel performance tables,
predictive model metrics, budget reallocations, and prioritized recommendations.
"""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.budget_optimizer import OptimizationResult
from src.metrics import format_currency, format_multiplier, format_percentage
from src.recommendation_engine import Recommendation


# Executive Brand Colors
NAVY = colors.HexColor("#232F3E")       # Amazon Navy
AMAZON_ORANGE = colors.HexColor("#FF9900")  # Amazon Orange
LIGHT_BG = colors.HexColor("#F8F9FA")
BORDER_COLOR = colors.HexColor("#DEE2E6")
TEXT_MUTED = colors.HexColor("#6C757D")
TEXT_DARK = colors.HexColor("#1A1A1A")
SUCCESS_COLOR = colors.HexColor("#28A745")
DANGER_COLOR = colors.HexColor("#DC3545")


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and print total page numbers: 'Page X of Y'."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, total_pages: int):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(TEXT_MUTED)

        # Running header (pages after page 1)
        if self._pageNumber > 1:
            self.drawString(54, 11 * inch - 36, "Amazon AI Marketing Budget Optimizer — Executive Report")
            self.drawRightString(8.5 * inch - 54, 11 * inch - 36, datetime.now().strftime("%B %d, %Y"))
            self.setStrokeColor(BORDER_COLOR)
            self.setLineWidth(0.5)
            self.line(54, 11 * inch - 42, 8.5 * inch - 54, 11 * inch - 42)

        # Running footer
        footer_text = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(8.5 * inch - 54, 34, footer_text)
        disclaimer = "Independent Case Study Demonstration. Confidential & Proprietary."
        self.drawString(54, 34, disclaimer)
        self.setStrokeColor(BORDER_COLOR)
        self.setLineWidth(0.5)
        self.line(54, 46, 8.5 * inch - 54, 46)

        self.restoreState()


def generate_pdf_report(
    exec_kpis: Dict[str, Any],
    channel_kpis: pd.DataFrame,
    underperforming_campaigns: pd.DataFrame,
    opt_result: Optional[OptimizationResult],
    recommendations: List[Recommendation],
    prediction_eval: Optional[Dict[str, Any]] = None,
    output_filename: Optional[str] = None,
) -> Tuple[str, bytes]:
    """Generate a publication-grade PDF executive report.

    Returns:
        output_filepath: Absolute file path where PDF was saved
        pdf_bytes: Raw bytes of the generated PDF document
    """
    reports_dir = Path(__file__).resolve().parent.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not output_filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"Amazon_Marketing_Budget_Optimization_Report_{timestamp}.pdf"

    output_path = reports_dir / output_filename

    # Build PDF buffer
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Custom Typography Styles
    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=NAVY,
    )
    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=TEXT_MUTED,
    )
    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=NAVY,
        spaceBefore=14,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=TEXT_DARK,
    )
    bold_body = ParagraphStyle(
        "BoldBody",
        parent=body_style,
        fontName="Helvetica-Bold",
    )
    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=9.5,
        textColor=TEXT_DARK,
    )
    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=table_cell,
        fontName="Helvetica-Bold",
    )
    table_cell_right = ParagraphStyle(
        "TableCellRight",
        parent=table_cell,
        alignment=2,  # Right align
    )
    badge_high = ParagraphStyle(
        "BadgeHigh",
        parent=table_cell,
        textColor=DANGER_COLOR,
        fontName="Helvetica-Bold",
    )
    badge_med = ParagraphStyle(
        "BadgeMed",
        parent=table_cell,
        textColor=AMAZON_ORANGE,
        fontName="Helvetica-Bold",
    )

    story = []

    # 1. Header Banner
    story.append(Paragraph("AMAZON MARKETING BUDGET OPTIMIZATION AGENT", title_style))
    story.append(Paragraph("AI-Driven Performance Analysis, Predictive Forecasting & Budget Reallocation Report", subtitle_style))
    story.append(Spacer(1, 4))
    gen_time_str = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    story.append(Paragraph(f"<b>Generated:</b> {gen_time_str} &nbsp;|&nbsp; <b>Evaluation Mode:</b> Enterprise Simulation &nbsp;|&nbsp; <b>Status:</b> Ready for Review", body_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=2, color=AMAZON_ORANGE, spaceBefore=2, spaceAfter=10))

    # 2. Executive Summary & Core KPIs Callout
    story.append(Paragraph("1. Executive Summary & Financial Highlights", h1_style))
    tot_spend = exec_kpis.get("total_spend", 0.0)
    tot_rev = exec_kpis.get("total_revenue", 0.0)
    roas = exec_kpis.get("overall_roas", 0.0)
    contrib = exec_kpis.get("marketing_contribution", 0.0)
    convs = exec_kpis.get("total_conversions", 0)

    summary_text = (
        f"This executive management report evaluates multi-channel advertising performance across Amazon Sponsored "
        f"Products, Sponsored Brands, Sponsored Display, Google Ads, Meta Ads, Email, and Affiliate channels. "
        f"Over the analyzed period, total marketing expenditure was <b>{format_currency(tot_spend)}</b>, generating "
        f"<b>{format_currency(tot_rev)}</b> in attributed revenue at a blended ROAS of <b>{format_multiplier(roas)}</b> "
        f"and <b>{format_currency(contrib)}</b> in estimated marketing contribution across <b>{convs:,}</b> customer conversions."
    )
    story.append(Paragraph(summary_text, body_style))
    story.append(Spacer(1, 8))

    # KPI Summary Cards Table
    kpi_box_data = [
        [
            Paragraph("<b>Total Spend</b>", table_cell),
            Paragraph("<b>Attributed Revenue</b>", table_cell),
            Paragraph("<b>Blended ROAS</b>", table_cell),
            Paragraph("<b>Marketing Contribution</b>", table_cell),
            Paragraph("<b>Total Conversions</b>", table_cell),
        ],
        [
            Paragraph(f"<b>{format_currency(tot_spend)}</b>", ParagraphStyle("BigKPI", parent=bold_body, fontSize=10, textColor=NAVY)),
            Paragraph(f"<b>{format_currency(tot_rev)}</b>", ParagraphStyle("BigKPI", parent=bold_body, fontSize=10, textColor=NAVY)),
            Paragraph(f"<b>{format_multiplier(roas)}</b>", ParagraphStyle("BigKPI", parent=bold_body, fontSize=10, textColor=SUCCESS_COLOR)),
            Paragraph(f"<b>{format_currency(contrib)}</b>", ParagraphStyle("BigKPI", parent=bold_body, fontSize=10, textColor=NAVY)),
            Paragraph(f"<b>{convs:,}</b>", ParagraphStyle("BigKPI", parent=bold_body, fontSize=10, textColor=NAVY)),
        ],
    ]
    kpi_box_table = Table(kpi_box_data, colWidths=[100, 105, 95, 115, 89])
    kpi_box_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
        ("BOX", (0, 0), (-1, -1), 1, BORDER_COLOR),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    story.append(kpi_box_table)
    story.append(Spacer(1, 10))

    # 3. Channel Performance Table
    story.append(Paragraph("2. Multi-Channel Advertising Performance Comparison", h1_style))
    chan_headers = [
        Paragraph("<b>Marketing Channel</b>", table_cell_bold),
        Paragraph("<b>Spend ($)</b>", table_cell_bold),
        Paragraph("<b>Revenue ($)</b>", table_cell_bold),
        Paragraph("<b>ROAS</b>", table_cell_bold),
        Paragraph("<b>CPA ($)</b>", table_cell_bold),
        Paragraph("<b>Contrib ($)</b>", table_cell_bold),
        Paragraph("<b>ROI (%)</b>", table_cell_bold),
    ]
    chan_rows = [chan_headers]
    for _, row in channel_kpis.iterrows():
        chan_rows.append([
            Paragraph(str(row["channel"]), table_cell),
            Paragraph(format_currency(row["marketing_spend"]), table_cell_right),
            Paragraph(format_currency(row["attributed_revenue"]), table_cell_right),
            Paragraph(format_multiplier(row.get("roas")), table_cell_right),
            Paragraph(format_currency(row.get("cpa")), table_cell_right),
            Paragraph(format_currency(row.get("marketing_contribution")), table_cell_right),
            Paragraph(format_percentage(row.get("roi")), table_cell_right),
        ])

    chan_table = Table(chan_rows, colWidths=[130, 60, 65, 45, 55, 75, 74])
    chan_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(chan_table)
    story.append(Spacer(1, 10))

    # 4. Underperforming Campaigns Highlight
    if not underperforming_campaigns.empty:
        story.append(Paragraph("3. Underperforming & Potentially Inefficient Campaigns", h1_style))
        story.append(Paragraph(
            "Campaigns identified as underperforming or inefficient based on sub-target ROAS, inflated CPA, "
            "and negative marketing contribution with sufficient statistical observation.",
            body_style,
        ))
        story.append(Spacer(1, 4))

        under_headers = [
            Paragraph("<b>Campaign Name</b>", table_cell_bold),
            Paragraph("<b>Channel</b>", table_cell_bold),
            Paragraph("<b>Spend</b>", table_cell_bold),
            Paragraph("<b>ROAS</b>", table_cell_bold),
            Paragraph("<b>CPA</b>", table_cell_bold),
            Paragraph("<b>Classification Reason</b>", table_cell_bold),
        ]
        under_rows = [under_headers]
        for _, row in underperforming_campaigns.head(5).iterrows():
            under_rows.append([
                Paragraph(str(row["campaign_name"]), table_cell),
                Paragraph(str(row["channel"]), table_cell),
                Paragraph(format_currency(row["marketing_spend"]), table_cell_right),
                Paragraph(format_multiplier(row.get("roas")), table_cell_right),
                Paragraph(format_currency(row.get("cpa")), table_cell_right),
                Paragraph(str(row.get("classification_reason", ""))[:120] + "...", table_cell),
            ])

        under_table = Table(under_rows, colWidths=[110, 85, 50, 40, 45, 174])
        under_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#495057")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
            ("GRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(under_table)
        story.append(Spacer(1, 10))

    # Page Break for Optimization and Strategic Recommendations
    story.append(PageBreak())

    # 5. Budget Optimization Results
    if opt_result is not None and not opt_result.channel_allocations.empty:
        story.append(Paragraph("4. Recommended Marketing Budget Reallocation (SciPy SLSQP)", h1_style))
        story.append(Paragraph(
            f"The AI optimizer evaluated non-linear diminishing marginal returns subject to the total budget of "
            f"<b>{format_currency(opt_result.total_budget)}</b>, isolating an experiment reserve of "
            f"<b>{format_currency(opt_result.experiment_reserve)} ({opt_result.reserve_pct:.1f}%)</b>. "
            f"Reallocation produces an estimated incremental marketing contribution of "
            f"<b>+{format_currency(opt_result.incremental_contribution)}</b>.",
            body_style,
        ))
        story.append(Spacer(1, 6))

        opt_headers = [
            Paragraph("<b>Channel</b>", table_cell_bold),
            Paragraph("<b>Current ($)</b>", table_cell_bold),
            Paragraph("<b>Recommended ($)</b>", table_cell_bold),
            Paragraph("<b>Shift ($)</b>", table_cell_bold),
            Paragraph("<b>Shift (%)</b>", table_cell_bold),
            Paragraph("<b>Exp. Revenue ($)</b>", table_cell_bold),
            Paragraph("<b>Exp. Contrib ($)</b>", table_cell_bold),
        ]
        opt_rows = [opt_headers]
        for _, row in opt_result.channel_allocations.iterrows():
            shift_val = row["budget_change"]
            shift_p = Paragraph(f"+{format_currency(shift_val)}" if shift_val > 0 else format_currency(shift_val), table_cell_right)
            opt_rows.append([
                Paragraph(str(row["channel"]), table_cell),
                Paragraph(format_currency(row["current_budget"]), table_cell_right),
                Paragraph(format_currency(row["recommended_budget"]), table_cell_right),
                shift_p,
                Paragraph(f"{row['change_pct']:+.1f}%", table_cell_right),
                Paragraph(format_currency(row["expected_revenue"]), table_cell_right),
                Paragraph(format_currency(row["expected_contribution"]), table_cell_right),
            ])

        opt_table = Table(opt_rows, colWidths=[124, 60, 65, 55, 50, 75, 75])
        opt_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
            ("GRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(opt_table)
        story.append(Spacer(1, 10))

    # 6. Strategic Recommendations
    story.append(Paragraph("5. Prioritized Actionable Recommendations", h1_style))
    rec_headers = [
        Paragraph("<b>Priority</b>", table_cell_bold),
        Paragraph("<b>Target</b>", table_cell_bold),
        Paragraph("<b>Category</b>", table_cell_bold),
        Paragraph("<b>Action & Rationale</b>", table_cell_bold),
        Paragraph("<b>Estimated Impact</b>", table_cell_bold),
    ]
    rec_rows = [rec_headers]
    for r in recommendations[:6]:
        p_badge = badge_high if r.priority == "HIGH" else badge_med
        rec_rows.append([
            p_badge,
            Paragraph(f"<b>{r.target_name}</b>", table_cell),
            Paragraph(r.category, table_cell),
            Paragraph(f"{r.recommended_action}<br/><font color='#6C757D'>{r.evidence}</font>", table_cell),
            Paragraph(r.estimated_budget_impact, table_cell),
        ])

    rec_table = Table(rec_rows, colWidths=[45, 95, 80, 204, 80])
    rec_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#343A40")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(rec_table)
    story.append(Spacer(1, 10))

    # 7. Model Evaluation & Governance
    if prediction_eval:
        story.append(Paragraph("6. Machine Learning Pipeline Governance", h1_style))
        ml_notes = (
            f"<b>Revenue Model:</b> {prediction_eval.get('model_name', 'Random Forest')} &nbsp;|&nbsp; "
            f"<b>Test Samples:</b> {prediction_eval.get('test_samples', 'N/A')} &nbsp;|&nbsp; "
            f"<b>MAE:</b> ${prediction_eval.get('mae', 'N/A')} &nbsp;|&nbsp; "
            f"<b>RMSE:</b> ${prediction_eval.get('rmse', 'N/A')} &nbsp;|&nbsp; "
            f"<b>R²:</b> {prediction_eval.get('r2', 'N/A')} "
            f"(Improvement vs historical baseline: +{prediction_eval.get('improvement_pct', 0.0)}%)"
        )
        story.append(Paragraph(ml_notes, body_style))
        story.append(Spacer(1, 6))

    # 8. Governance, Human Approval & Disclaimer Box
    governance_box = [
        [
            Paragraph(
                "<b>GOVERNANCE & HUMAN APPROVAL NOTICE:</b><br/>"
                "This report provides automated decision-support recommendations generated from machine learning "
                "models and constrained mathematical optimization. It does not automatically modify active Amazon Ads "
                "or external ad network campaigns. All budget shifts and bid changes require explicit review and authorization "
                "by authorized marketing leadership before live execution.<br/>"
                "<b>Disclaimer:</b> Independent demonstration project utilizing synthetic e-commerce data. Not an official "
                "Amazon application.",
                ParagraphStyle("GovNotice", parent=body_style, fontSize=7.5, leading=10, textColor=NAVY),
            )
        ]
    ]
    gov_table = Table(governance_box, colWidths=[504])
    gov_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFF3CD")),
        ("BOX", (0, 0), (-1, -1), 1, AMAZON_ORANGE),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(KeepTogether(gov_table))

    # Build the document
    doc.build(story, canvasmaker=NumberedCanvas)
    pdf_bytes = buffer.getvalue()

    # Save to disk
    with open(output_path, "wb") as f:
        f.write(pdf_bytes)

    return str(output_path), pdf_bytes
