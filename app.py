import streamlit as st
import tempfile
import pandas as pd
from main_pipeline import process_video

st.set_page_config(page_title="Helmet Violation Detector", layout="wide")

st.title("🪖 AI Helmet Violation Detection")

uploaded_file = st.file_uploader("Upload Traffic Video", type=["mp4","avi","mov"])

if uploaded_file:
    st.success("Video uploaded!")

    temp_video = tempfile.NamedTemporaryFile(delete=False)
    temp_video.write(uploaded_file.read())

    if st.button("🚀 Analyze Video"):
        with st.spinner("Processing video..."):
            output_video, violations = process_video(temp_video.name)

        st.video(output_video)

        st.subheader("🚨 Violations Detected")
        df = pd.DataFrame(violations)
        st.dataframe(df)
