"""
recommendations.py
Maps predicted DR severity class to patient guidance (Do's and Don'ts)
and urgency level. Used by inference.py / app.py to enrich the raw
model prediction with actionable advice.
"""

RECOMMENDATIONS = {
    "No DR": {
        "urgency": "Routine",
        "color": "#2ecc71",  # green
        "dos": [
            "Continue annual eye screening",
            "Maintain healthy blood sugar (HbA1c) levels",
            "Keep blood pressure under control",
            "Maintain a balanced diet and regular exercise",
        ],
        "donts": [
            "Don't skip yearly checkups even if vision feels fine",
            "Don't ignore family history of diabetes or eye disease",
        ],
    },
    "Mild": {
        "urgency": "Low",
        "color": "#a3d900",  # yellow-green
        "dos": [
            "Get re-screened every 6–12 months",
            "Keep blood sugar and blood pressure tightly controlled",
            "Monitor for any new vision changes and report them promptly",
        ],
        "donts": [
            "Don't delay the next follow-up screening",
            "Don't self-medicate or self-diagnose without consulting a doctor",
        ],
    },
    "Moderate": {
        "urgency": "Moderate - consult within 3-6 months",
        "color": "#f1c40f",  # yellow/amber
        "dos": [
            "Consult an ophthalmologist within 3–6 months",
            "Strictly manage blood sugar and HbA1c levels",
            "Monitor vision closely for any sudden changes",
        ],
        "donts": [
            "Don't wait for visible symptoms before seeking care",
            "Don't ignore blurred, fluctuating, or spotty vision",
        ],
    },
    "Severe": {
        "urgency": "High - urgent referral (within weeks)",
        "color": "#e67e22",  # orange
        "dos": [
            "Seek an ophthalmologist consultation urgently (within a few weeks)",
            "Get a comprehensive dilated eye examination",
            "Follow a strict diabetes management plan under medical supervision",
        ],
        "donts": [
            "Don't delay referral — this stage can progress quickly",
            "Don't rely on this screening alone — it needs clinical confirmation",
        ],
    },
    "Proliferative DR": {
        "urgency": "Critical - immediate specialist care",
        "color": "#e74c3c",  # red
        "dos": [
            "Seek immediate specialist / emergency eye care",
            "Discuss treatment options (laser therapy, anti-VEGF injections) with a retina specialist",
            "Get an urgent comprehensive eye exam without delay",
        ],
        "donts": [
            "Don't delay care — this stage carries real risk of vision loss",
            "Don't attempt to self-manage or wait and watch at this stage",
        ],
    },
}


def get_recommendation(predicted_class: str) -> dict:
    """
    Returns the recommendation dict for a given predicted class label.
    Falls back to a safe default if the label doesn't match exactly
    (e.g. due to casing or naming differences from your model's class list).
    """
    if predicted_class in RECOMMENDATIONS:
        return RECOMMENDATIONS[predicted_class]

    # fallback: try case-insensitive match
    for key in RECOMMENDATIONS:
        if key.lower() == predicted_class.lower():
            return RECOMMENDATIONS[key]

    # last-resort default (should not normally trigger)
    return {
        "urgency": "Unknown — please consult a doctor",
        "color": "#7f8c8d",
        "dos": ["Consult a healthcare professional for proper evaluation"],
        "donts": ["Don't rely solely on this result"],
    }