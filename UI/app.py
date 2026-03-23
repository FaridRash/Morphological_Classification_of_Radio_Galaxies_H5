import streamlit as st
import requests

API_URL = "https://radio-galaxy-api-783206752653.europe-west1.run.app"

st.title("Radio Galaxy Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, use_container_width=True)

if uploaded_file is not None and st.button("Predict"):
    with st.spinner("Sending to API..."):
        response = requests.post(
            API_URL,
            files={"file": uploaded_file.getvalue()}
        )

    st.write("Status:", response.status_code)

    if response.status_code == 200:
        st.json(response.json())
    else:
        st.error(response.text)