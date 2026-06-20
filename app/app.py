import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="SMS Spam Classifier", page_icon="📩", layout="centered")

@st.cache_resource
def load_model():
    return pipeline("text-classification", model="Goodmotion/spam-mail-classifier")

classifier = load_model()

st.title("📩 SMS Spam Classifier")
st.write(
    "Enter a message below and the model will classify it as **SPAM** or **HAM**."
)

example_messages = {
    "Spam example": "Congratulations! You have won a free vacation. Call now to claim your prize.",
    "Ham example": "Hi, I’ll be there in 10 minutes. Please wait for me at the entrance."
}

selected_example = st.selectbox(
    "Optional: load an example message",
    options=["None"] + list(example_messages.keys())
)

default_text = ""
if selected_example != "None":
    default_text = example_messages[selected_example]

user_text = st.text_area("Message text", value=default_text, height=180)

if st.button("Classify message"):
    if not user_text.strip():
        st.warning("Please enter a message before classifying.")
    else:
        result = classifier(user_text)[0]
        raw_label = result["label"]
        score = float(result["score"])

        # Normalize output to SPAM / HAM for display
        label_upper = raw_label.upper()
        if "SPAM" in label_upper:
            final_label = "SPAM"
        elif "HAM" in label_upper:
            final_label = "HAM"
        else:
            # Fallback in case the model returns something like LABEL_0 / LABEL_1
            final_label = raw_label

        if final_label == "SPAM":
            st.error(f"Prediction: {final_label}")
        else:
            st.success(f"Prediction: {final_label}")

        st.metric("Confidence", f"{score:.2%}")

        st.subheader("Raw model output")
        st.json(result)

st.markdown("---")
st.caption("Model: Goodmotion/spam-mail-classifier")
