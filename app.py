"""
Streamlit UI for the Email/SMS Spam Detection system.

Usage:
    streamlit run app.py
"""

import streamlit as st

# ---------- Page configuration ----------
st.set_page_config(
    page_title="Spam Email Classificator",
    page_icon="📧",
    layout="centered",
)

# ---------- Lazy-import predict (loads model once) ----------
from predict import predict  # noqa: E402  – imported after set_page_config

# ---------- Sidebar ----------
with st.sidebar:
    st.title("ℹ️ About")
    st.markdown(
        """
        This application uses a **Deep-Learning** model to classify
        Spam emails or fake Emails

        **Stack**: Streamlit · scikit-learn · NLTK

        **How it works**
        1. Text is lowered, tokenised, cleaned of stop-words and stemmed.
        2. A TF-IDF vectoriser converts it to features.
        3. The trained classifier predicts the label with a confidence score.
        """
    )
    st.divider()
    st.caption("Built by Abhiram. M")

# ---------- Main content ----------
st.title("📧 Spam Email Classificator ")
st.markdown("Enter an email below and click **Predict** to check if it is spam.")

input_text = st.text_area(
    "Message",
    height=180,
    placeholder="Type or paste the message here …",
)

if st.button("🔍 Predict", use_container_width=True):
    # ---- Input validation ----
    if not input_text or not input_text.strip():
        st.warning("⚠️ Please enter some text before predicting.")
    else:
        try:
            result = predict(input_text)
            label = result["label"]
            confidence = result["confidence"]

            # ---- Display result ----
            if label == "spam":
                st.error(f"🚨 **SPAM** detected!")
            else:
                st.success(f"✅ This message looks **safe**.")

            # Confidence bar
            st.markdown(f"**Confidence:** {confidence:.1%}")
            st.progress(confidence)

        except ValueError as exc:
            st.warning(f"⚠️ {exc}")
        except Exception as exc:
            st.error(f"❌ An unexpected error occurred: {exc}")
