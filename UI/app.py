import streamlit as st
import requests

API_URL = "https://radio-galaxy-api-783206752653.europe-west1.run.app/"

st.set_page_config(page_title="Radio Galaxy Classifier", layout="centered")

st.title("🔭 Radio Galaxy Classifier")
st.markdown("Upload a radio galaxy image to classify its morphology")

uploaded_file = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(uploaded_file, use_container_width=True)

    if st.button("Predict"):
        with st.spinner("Analyzing..."):
            image_bytes = uploaded_file.getvalue()

            # detect correct content type
            filename = uploaded_file.name.lower()
            content_type = "image/png" if filename.endswith(".png") else "image/jpeg"

            response = requests.post(
                API_URL,
                data=image_bytes,
                headers={"Content-Type": content_type}
            )

        if response.status_code == 200:
            result = response.json()

            label = result["predicted_label"]
            probs = result["probabilities"]

            confidence = max(probs)

            st.success(f"Prediction: {label}")
            st.metric("Confidence", f"{confidence*100:.2f}%")

            st.subheader("Class Probabilities")
            st.bar_chart(probs)

        else:
            st.error("Prediction failed. Try another image.")