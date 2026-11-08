from __future__ import annotations

from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


def dossier_to_pdf(title: str, dossier: dict) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, title=title)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("SuppleScope Research Dossier", styles["Title"]),
        Paragraph("Look Beyond the Label. Discover the Evidence.", styles["Italic"]),
        Spacer(1, 12),
        Paragraph(
            "This report organizes retrieved sources. It is not a medical diagnosis, "
            "prescription, or custom-model verification verdict.",
            styles["Normal"],
        ),
        Spacer(1, 12),
    ]
    identity = dossier.get("product_identity") or {}
    story.append(Paragraph(f"Product: {identity.get('product_name') or 'Unknown'}", styles["Heading2"]))
    story.append(Paragraph(f"Identity status: {identity.get('identity_status')}", styles["Normal"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Uncertainties", styles["Heading2"]))
    for item in dossier.get("uncertainties") or ["None recorded"]:
        story.append(Paragraph(f"• {item}", styles["Normal"]))
    story.append(Paragraph("Sources", styles["Heading2"]))
    for source in dossier.get("sources") or []:
        line = f"{source.get('title') or 'Untitled'} — {source.get('url') or 'no URL'}"
        story.append(Paragraph(line.replace("&", "&amp;"), styles["Normal"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(dossier.get("medical_disclaimer") or "", styles["Italic"]))
    doc.build(story)
    return buffer.getvalue()
