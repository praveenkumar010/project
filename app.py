import streamlit as st
import pandas as pd
import tempfile
import os
from main_pipeline import process_video

st.set_page_config(page_title="Helmet Violation Detection", layout="wide")

st.title("🚨 Helmet Violation Detection System")
st.write("Upload a traffic video and detect helmet violations automatically.")

# Upload video
uploaded_video = st.file_uploader("📤 Upload Traffic Video", type=["mp4", "avi", "mov"])

if uploaded_video is not None:

    st.subheader("📹 Uploaded Video")
    st.video(uploaded_video)

    if st.button("▶️ Run Detection"):

        # 🔥 Save uploaded video to temp file (IMPORTANT for cloud)
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        input_video_path = tfile.name

        st.info("Processing video... please wait ⏳")

        # 🚀 Run pipeline
        try:
            output_video_path, violations = process_video(input_video_path)

            st.success("✅ Processing Completed!")

            # 🎬 Show output video
            if os.path.exists(output_video_path):
                st.subheader("🎬 Output Video")
                video_file = open(output_video_path, 'rb')
                video_bytes = video_file.read()
                st.video(video_bytes)
            else:
                st.error("Output video not found ❌")

            # 📊 Show violation table
            st.subheader("📋 Violations Detected")

            if len(violations) > 0:
                df = pd.DataFrame(violations, columns=["Vehicle ID", "Violation"])
                st.dataframe(df, use_container_width=True)
            else:
                st.success("🎉 No violations detected!")

        except Exception as e:
            st.error(f"Something went wrong ❌\n\n{e}")
