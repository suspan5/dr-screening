"""
report_generator.py
Generates a downloadable PDF report summarizing a single DR
screening result: prediction, confidence, Grad-CAM explanation,
and severity-based recommendations.

Requires: pip install fpdf2

NOTE: This version deliberately avoids fpdf2's multi_cell() for
paragraph wrapping, because certain fpdf2 versions raise
"Not enough horizontal space to render a single character" in
edge cases with its internal line-break algorithm. Instead we
wrap text manually using get_string_width() and print line-by-line
with cell(), which is more predictable.
"""

import os
import tempfile
from datetime import datetime

import numpy as np
from PIL import Image
from fpdf import FPDF


def _safe_text(text) -> str:
    """
    Sanitizes text for FPDF's core (non-unicode) fonts.
    Replaces common typographic characters with plain ASCII equivalents,
    then strips anything else that isn't Latin-1 encodable.
    """
    text = str(text)

    replacements = {
        "\u2014": "-", "\u2013": "-",
        "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2026": "...", "\u2022": "-",
        "\u2192": "->", "\u00b1": "+/-",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)

    return text.encode("latin-1", "ignore").decode("latin-1")


def _print_wrapped(pdf, text, line_height=6):
    """
    Manually wraps `text` to fit the printable page width and prints
    it line by line using cell() — avoids multi_cell()'s buggy
    internal line-break algorithm entirely.
    """
    text = _safe_text(text)
    max_width = pdf.w - pdf.l_margin - pdf.r_margin

    words = text.split(" ")
    line = ""

    for word in words:
        candidate = (line + " " + word).strip() if line else word

        if pdf.get_string_width(candidate) <= max_width:
            line = candidate
        else:
            if line:
                pdf.set_x(pdf.l_margin)
                pdf.cell(0, line_height, line, ln=True)
            if pdf.get_string_width(word) > max_width:
                chunk = ""
                for ch in word:
                    if pdf.get_string_width(chunk + ch) <= max_width:
                        chunk += ch
                    else:
                        pdf.set_x(pdf.l_margin)
                        pdf.cell(0, line_height, chunk, ln=True)
                        chunk = ch
                line = chunk
            else:
                line = word

    if line:
        pdf.set_x(pdf.l_margin)
        pdf.cell(0, line_height, line, ln=True)


def _save_temp_image(image_obj, suffix=".png"):
    """
    Saves an image to a temp file and returns its path.
    Accepts either a PIL.Image, a numpy array, or a file path string.
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name
    tmp.close()

    if isinstance(image_obj, Image.Image):
        image_obj.save(tmp_path)

    elif isinstance(image_obj, np.ndarray):
        arr = image_obj
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(tmp_path)

    elif isinstance(image_obj, str) and os.path.exists(image_obj):
        Image.open(image_obj).save(tmp_path)

    else:
        raise TypeError(
            f"Unsupported image type for PDF report: {type(image_obj)}. "
            "Expected PIL.Image, numpy.ndarray, or a valid file path."
        )

    return tmp_path


def generate_report(
    original_image,
    gradcam_image,
    prediction: str,
    confidence: float,
    severity: str,
    referral: str,
    urgency: str,
    recommendation: dict,
    probabilities: dict,
    output_path: str = "outputs/reports/DR_Screening_Report.pdf",
) -> str:
    """
    Builds a PDF report and returns the path to the saved file.
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    orig_path = _save_temp_image(original_image)
    cam_path = _save_temp_image(gradcam_image)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ---------------- Header ----------------
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(20, 20, 20)
    pdf.cell(0, 12, "Diabetic Retinopathy Screening Report", ln=True, align="C")

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(
        0, 8,
        _safe_text(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"),
        ln=True, align="C"
    )
    pdf.ln(4)
    pdf.set_x(pdf.l_margin)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(8)

    # ---------------- Prediction summary ----------------
    pdf.set_x(pdf.l_margin)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, _safe_text(f"Prediction: {prediction}"), ln=True)

    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, _safe_text(f"Confidence: {confidence * 100:.2f}%"), ln=True)
    pdf.set_x(pdf.l_margin)
    pdf.cell(0, 7, _safe_text(f"Severity: {severity}"), ln=True)
    pdf.set_x(pdf.l_margin)
    pdf.cell(0, 7, _safe_text(f"Urgency: {urgency}"), ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "I", 10)
    _print_wrapped(pdf, f"Recommendation: {referral}", line_height=6)
    pdf.ln(4)

    # ---------------- Images side by side ----------------
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Retinal Image & AI Explanation (Grad-CAM)", ln=True)
    pdf.ln(2)

    y_before_images = pdf.get_y()
    img_width = 90

    pdf.image(orig_path, x=10, y=y_before_images, w=img_width)
    pdf.image(cam_path, x=110, y=y_before_images, w=img_width)

    pdf.set_xy(pdf.l_margin, y_before_images + img_width * 0.75 + 8)

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(90, 90, 90)
    pdf.cell(95, 5, "Original Image", align="C")
    pdf.cell(95, 5, "Grad-CAM Explanation", align="C", ln=True)
    pdf.set_x(pdf.l_margin)
    pdf.ln(6)

    # ---------------- Class probabilities ----------------
    pdf.set_x(pdf.l_margin)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Class Probabilities", ln=True)
    pdf.set_font("Helvetica", "", 10)

    for class_name, prob in probabilities.items():
        pdf.set_x(pdf.l_margin)
        pdf.cell(0, 6, _safe_text(f"{class_name}: {prob * 100:.2f}%"), ln=True)

    pdf.ln(4)

    # ---------------- Do's and Don'ts ----------------
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Guidance", ln=True)

    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 120, 30)
    pdf.cell(0, 6, "Do's:", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(0, 0, 0)
    for item in recommendation.get("dos", []):
        _print_wrapped(pdf, f"+ {item}", line_height=6)

    pdf.ln(2)
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(150, 30, 30)
    pdf.cell(0, 6, "Don'ts:", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(0, 0, 0)
    for item in recommendation.get("donts", []):
        _print_wrapped(pdf, f"- {item}", line_height=6)

    pdf.ln(6)

    # ---------------- Disclaimer ----------------
    pdf.set_x(pdf.l_margin)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(100, 100, 100)
    _print_wrapped(
        pdf,
        "Disclaimer: This report is generated by an AI-assisted screening "
        "tool intended for decision support only. It is not a substitute "
        "for professional medical diagnosis. Please consult a qualified "
        "ophthalmologist for clinical evaluation and confirmation.",
        line_height=5,
    )

    pdf.output(output_path)

    os.remove(orig_path)
    os.remove(cam_path)

    return output_path