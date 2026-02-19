import streamlit as st
import pandas as pd
from main_pipeline import process_video
import tempfile

st.set_page_config(layout="wide")
st.title("🚨 Helmet Violation Detection System")

uploaded_video = st.file_uploader("Upload Traffic Video", type=["mp4","mov","avi"])

if uploaded_video is not None:

    st.info("Processing video... please wait ⏳")

    # save temp video
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_video.read())

    # run detection
    output_video, violations = process_video(temp_file.name)

    st.success("Processing Completed ✅")

    st.subheader("🎬 Processed Video")
    video_file = open(output_video, 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)

    st.subheader("📋 Violations Detected")

    if len(violations) == 0:
        st.success("No Violations Detected 🎉")
    else:
        df = pd.DataFrame(violations, columns=["Vehicle ID","Violation"])
        st.dataframe(df, use_container_width=True)
