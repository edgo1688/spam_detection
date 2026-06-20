import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="SMS Spam Classifier", page_icon="📩", layout="centered")

@st.cache_resource
def load_model():
    return pipeline("text-classification", model="Goodmotion/spam-mail-classifier")


def normalize_prediction_label(
    raw_label: str,
    id2label: dict | None = None,
    label2id: dict | None = None,
) -> str:
    label = str(raw_label)

    # Try to map LABEL_<n> tokens using id2label. Be robust to keys
    # being ints or strings and to layered mappings (e.g. mapping to
    # another LABEL_<n> token).
    if id2label is not None:
        for _ in range(3):  # small loop to resolve nested LABEL_ mappings
            if not label.upper().startswith("LABEL_"):
                break
            label_index = label.split("_")[-1]
            if not label_index.isdigit():
                break

            idx_int = int(label_index)
            mapped = None
            # Try integer key
            try:
                mapped = id2label.get(idx_int)
            except Exception:
                mapped = None

            # Try string key forms if integer lookup failed
            if mapped is None:
                mapped = id2label.get(str(idx_int)) if isinstance(id2label, dict) else None
            if mapped is None:
                # Some configs use string keys that are not plain ints
                mapped = id2label.get(label_index) if isinstance(id2label, dict) else None

            if mapped is None:
                break

            label = str(mapped)

    # If we still have a LABEL_<n> token, try to invert a provided
    # `label2id` mapping (name -> id) to get a human-readable name.
    if label_upper := label.upper():
        pass
    if label_upper.find("LABEL_") == 0 and label2id is not None:
        try:
            inv = {int(v): k for k, v in label2id.items()}
            label_index = None
            if raw_label.upper().startswith("LABEL_"):
                idx = raw_label.split("_")[-1]
                if idx.isdigit():
                    label_index = int(idx)
            if label_index is not None:
                mapped_name = inv.get(label_index)
                if mapped_name:
                    label = str(mapped_name)
        except Exception:
            # If inversion fails, just continue with existing label
            pass

    label_upper = label.upper()
    if "SPAM" in label_upper:
        return "SPAM"
    if "HAM" in label_upper:
        return "HAM"
    return label


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

        final_label = normalize_prediction_label(
            raw_label,
            getattr(classifier.model.config, "id2label", None),
            getattr(classifier.model.config, "label2id", None),
        )
        # If normalization didn't resolve (e.g. still LABEL_1), try a
        # lightweight probe to map model label tokens to human labels.
        def _resolve_label_from_model(pipe, raw_lbl: str) -> str | None:
            # If we've already computed a resolved mapping, reuse it.
            if hasattr(pipe, "_resolved_id2label") and pipe._resolved_id2label:
                idx = None
                if str(raw_lbl).upper().startswith("LABEL_"):
                    tail = str(raw_lbl).split("_")[-1]
                    if tail.isdigit():
                        idx = int(tail)
                if idx is not None:
                    return pipe._resolved_id2label.get(idx)
                return None

            # Probe model with a spam and a ham example to observe labels.
            try:
                spam_out = pipe("Congratulations! You have won a free vacation. Call now to claim your prize.")[0]["label"]
                ham_out = pipe("Hi, I’ll be there in 10 minutes. Please wait for me at the entrance.")[0]["label"]

                def lbl_idx(l):
                    if str(l).upper().startswith("LABEL_"):
                        t = str(l).split("_")[-1]
                        if t.isdigit():
                            return int(t)
                    return None

                s_idx = lbl_idx(spam_out)
                h_idx = lbl_idx(ham_out)
                mapping = {}
                if s_idx is not None:
                    mapping[s_idx] = "SPAM"
                if h_idx is not None:
                    mapping[h_idx] = "HAM"

                # cache mapping on the pipeline object for future calls
                pipe._resolved_id2label = mapping

                if str(raw_lbl).upper().startswith("LABEL_"):
                    tail = str(raw_lbl).split("_")[-1]
                    if tail.isdigit():
                        return mapping.get(int(tail))
            except Exception:
                pass
            return None

        resolved_by_probe = None
        if not (final_label == "SPAM" or final_label == "HAM"):
            resolved_by_probe = _resolve_label_from_model(classifier, raw_label)
            if resolved_by_probe:
                final_label = resolved_by_probe

        if final_label == "SPAM":
            st.error(f"Prediction: {final_label}")
        else:
            st.success(f"Prediction: {final_label}")

        st.metric("Confidence", f"{score:.2%}")

        st.subheader("Raw model output")
        st.json(result)

st.markdown("---")
st.caption("Model: Goodmotion/spam-mail-classifier")
