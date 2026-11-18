import streamlit as st

# streamlit run app/ui.py ya project root se, dono tarah chale
try:
    from app.vision_utils import ask_video, describe_video
except ModuleNotFoundError:
    from vision_utils import ask_video, describe_video

st.set_page_config(page_title="YouTube Video RAG Extractor", layout="wide")

st.title("📹 YouTube Video RAG Extractor")
st.write("Apna YouTube video link do aur uske content se kuch bhi sawaal pucho!")

# Sidebar: video URL + ek baar analyze
with st.container():
    st.header("1. Video URL")
    video_url = st.text_input("YouTube Video URL:")

    if st.button("Analyze Video (sirf ek baar)"):
        if video_url:
            bar = st.progress(0, text="Shuru ho raha hai...")

            def on_progress(done, total):
                bar.progress(done / total, text=f"{done}/{total} hisse ho gaye")

            try:
                describe_video(video_url, progress=on_progress)
                bar.progress(1.0, text="Ready!")
                st.success("Ab sawaal turant milenge.")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Pehle URL daalo.")

    st.caption("Pehli baar Analyze dabao. Uske baad us video ke sawaal tez aayenge.")

# Main chat interface
st.header("2. Chat with Video")
query = st.text_input("Video ke bare mein kya janna chahte ho?")

if query:
    if not video_url:
        st.warning("Pehle upar YouTube URL daalo.")
    else:
        st.markdown("### Answer:")
        try:
            with st.spinner("Jawab ban raha hai..."):
                st.write_stream(ask_video(video_url, query))
        except Exception as e:
            st.error(f"Error: {e}")