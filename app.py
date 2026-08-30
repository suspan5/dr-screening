import streamlit as st
from PIL import Image
import numpy as np

from src.inference import predict_image
from src.recommendations import get_recommendation
from src.report_generator import generate_report

# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="DR Screening System",
    page_icon="🩺",
    layout="wide"
)


# ============================================================
# Custom styling
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
        margin-bottom: 30px;
    }

    .result-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ddd;
        margin-bottom: 15px;
    }

    .result-value {
        font-size: 28px;
        font-weight: 700;
    }

    .disclaimer {
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #ddd;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# Header
# ============================================================

st.markdown(
    '<div class="main-title">🩺 Diabetic Retinopathy Screening</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-assisted retinal image screening with explainable predictions'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.header("About")

    st.write(
        """
        This prototype uses a **MobileNetV2** deep-learning model
        trained to classify retinal fundus images into five
        diabetic retinopathy severity categories.
        """
    )

    st.divider()

    st.subheader("Classes")

    st.write("0 — No DR")
    st.write("1 — Mild")
    st.write("2 — Moderate")
    st.write("3 — Severe")
    st.write("4 — Proliferative DR")

    st.divider()

    st.caption(
        "For screening and decision support only."
    )


# ============================================================
# Upload section
# ============================================================

st.header("1. Upload Retinal Image")

uploaded_file = st.file_uploader(
    "Choose a retinal fundus image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# Image preview
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")

    st.image(
        image,
        caption="Retinal fundus image",
        width="stretch"
    )

    analyze = st.button(
        "🔍 Analyze Image",
        type="primary",
        use_container_width=True
    )

    if analyze:

        with st.spinner(
            "Analyzing retinal image..."
        ):

            result = predict_image(image)

        st.success("Analysis completed successfully.")


        # ====================================================
        # Results
        # ====================================================

        st.header("2. Screening Result")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.markdown(
                '<div class="result-card">'
                '<div>Prediction</div>'
                f'<div class="result-value">'
                f'{result["prediction"]}'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )

        with col2:

            st.markdown(
                '<div class="result-card">'
                '<div>Confidence</div>'
                f'<div class="result-value">'
                f'{result["confidence"] * 100:.2f}%'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )

        with col3:

            st.markdown(
                '<div class="result-card">'
                '<div>Urgency</div>'
                f'<div class="result-value">'
                f'{result["urgency"]}'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )


        # ====================================================
        # Severity / referral
        # ====================================================

        st.subheader("Screening Guidance")

        st.write(
            f"**Severity:** {result['severity']}"
        )

        st.write(
            f"**Recommendation:** {result['referral']}"
        )

                # ====================================================
        # Do's and Don'ts
        # ====================================================

        rec = get_recommendation(result["prediction"])

        st.markdown(
            f'<div class="result-card" style="border-left: 6px solid {rec["color"]};">'
            f'<b>Urgency Level:</b> {rec["urgency"]}'
            '</div>',
            unsafe_allow_html=True
        )

        col_do, col_dont = st.columns(2)

        with col_do:
            st.markdown("**✅ Do's**")
            for item in rec["dos"]:
                st.write(f"- {item}")

        with col_dont:
            st.markdown("**❌ Don'ts**")
            for item in rec["donts"]:
                st.write(f"- {item}")


        # ====================================================
        # Class probabilities
        # ====================================================

        st.subheader("Class Probabilities")

        probabilities = result["probabilities"]

        for class_name, probability in probabilities.items():

            st.write(
                f"**{class_name}** — "
                f"{probability * 100:.2f}%"
            )

            st.progress(
                float(probability)
            )


        # ====================================================
        # Grad-CAM
        # ====================================================

        st.header("3. AI Explanation — Grad-CAM")

        st.write(
            "The heatmap highlights image regions that "
            "contributed to the model's prediction."
        )

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                image,
                caption="Original retinal image",
                width="stretch"
            )

        with col2:

            st.image(
                result["gradcam"],
                caption="Grad-CAM explanation",
                width="stretch"
            )

                    # ====================================================
        # PDF Report Download
        # ====================================================

        st.header("4. Download Report")

        report_path = generate_report(
            original_image=image,
            gradcam_image=result["gradcam"],
            prediction=result["prediction"],
            confidence=result["confidence"],
            severity=result["severity"],
            referral=result["referral"],
            urgency=result["urgency"],
            recommendation=rec,
            probabilities=result["probabilities"],
        )

        with open(report_path, "rb") as f:
            st.download_button(
                label="📄 Download Full Screening Report (PDF)",
                data=f,
                file_name="DR_Screening_Report.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
            )


        # ====================================================
        # Disclaimer
        # ====================================================

        st.divider()

        st.markdown(
            '<div class="disclaimer">'
            '<b>⚠️ Important:</b> This system is intended for '
            'screening and decision support. It does not replace '
            'professional medical diagnosis or ophthalmologist '
            'evaluation.'
            '</div>',
            unsafe_allow_html=True
        )

else:

    st.info(
        "Upload a retinal fundus image to begin screening."
    )