import streamlit as st

# streamlit run app/ui.py ya project root se, dono tarah chale
try:
    from app.vision_utils import ask_video, describe_video
except ModuleNotFoundError:
    from vision_utils import ask_video, describe_video

st.set_page_config(page_title="YouTube Video RAG Extractor", layout="wide")

st.title("📹 YouTube Video RAG Extractor")
st.write("Paste a YouTube link and ask questions about the video.")

# Sidebar: video URL + ek baar analyze
with st.container():
    st.header("1. Video URL")
    video_url = st.text_input("YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")

    if st.button("Analyze Video"):
        if video_url:
            bar = st.progress(0, text="Starting...")

            def on_progress(done, total):
                bar.progress(done / total, text=f"{done}/{total} segments processed")

            try:
                describe_video(video_url, progress=on_progress)
                bar.progress(1.0, text="Ready!")
                st.success("Done! You can now ask questions about this video.")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please enter a YouTube URL first.")

    st.caption("Click Analyze first. After that, questions about this video are answered quickly.")

# Main chat interface
st.header("2. Chat with Video")
query = st.text_input("What would you like to know about the video?")

if query:
    if not video_url:
        st.warning("Please enter a YouTube URL above first.")
    else:
        st.markdown("### Answer:")
        try:
            with st.spinner("Generating answer..."):
                st.write_stream(ask_video(video_url, query))
        except Exception as e:
            st.error(f"Error: {e}")