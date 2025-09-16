import streamlit as st
from PIL import Image
import io
import os
import base64
import time

from models import (
    get_openai_response,
    get_gemini_response,
    get_groq_response,
    multimodal_response,
    gemini_multimodal,
    generate_image,
    transcribe_audio,
    speak_text
)

st.set_page_config(page_title="ChatBot", layout="wide")

theme_choice = st.sidebar.radio("Theme", options=["Light", "Dark"])
if theme_choice == "Dark":
    dark_css = """
    <style>
    body, [class^="st-"], [class*="st-"] {
        background-color: #0e1117 !important;
        color: #e6edf3 !important;
    }
    [data-testid="stSidebar"] { background-color: #0b0f14 !important; color: #e6edf3; }
    .stButton>button { background-color: #1f2937; color: #e6edf3; }
    .chat-message.user { background-color: #2b2f36 !important; color: #fff !important; }
    </style>
    """
    st.markdown(dark_css, unsafe_allow_html=True)
else:
    light_css = """
    <style>
    body, [class^="st-"], [class*="st-"] {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    [data-testid="stSidebar"] { background-color: #f8f9fa !important; color: #000000; }
    </style>
    """
    st.markdown(light_css, unsafe_allow_html=True)

chat_bubble_style = """
<style>
.chat-message {
    padding: 10px 15px;
    border-radius: 18px;
    margin: 8px 0;
    max-width: 75%;
    word-wrap: break-word;
    font-size: 15px;
}
.chat-message.user {
    background-color: #d3d3d3;
    color: #000;
    margin-left: auto;
    text-align: right;
}
.chat-message.assistant {
    background-color: #1e90ff;
    color: #fff;
    margin-right: auto;
    text-align: left;
}
</style>
"""
st.markdown(chat_bubble_style, unsafe_allow_html=True)

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {"Chat 1": []}
if "current_session" not in st.session_state:
    st.session_state.current_session = "Chat 1"
if "uploaded_images" not in st.session_state:
    st.session_state.uploaded_images = {}

model_choice = st.sidebar.selectbox("Choose AI Model", ["OpenAI (text & image)", "Gemini (image & text)", "Groq (simulated)"])
st.sidebar.markdown("---")
st.sidebar.subheader("Chats")

for name in list(st.session_state.chat_sessions.keys()):
    if st.sidebar.button(name, key=f"chat_{name}", use_container_width=True):
        st.session_state.current_session = name

if st.sidebar.button("➕ New Chat", use_container_width=True):
    chat_count = len(st.session_state.chat_sessions) + 1
    new_chat_name = f"Chat {chat_count}"
    st.session_state.chat_sessions[new_chat_name] = []
    st.session_state.current_session = new_chat_name

if len(st.session_state.chat_sessions) > 1:
    if st.sidebar.button("🗑️ Delete Chat", use_container_width=True):
        del st.session_state.chat_sessions[st.session_state.current_session]
        st.session_state.current_session = list(st.session_state.chat_sessions.keys())[0]

st.title("Chatbot")

def render_chat_history():
    hist = st.session_state.chat_sessions[st.session_state.current_session]
    for message in hist:
        role, content = message["role"], message["content"]
        st.markdown(f'<div class="chat-message {role}">{content}</div>', unsafe_allow_html=True)

render_chat_history()

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader(
    "Upload Audio or Image",
    type=["wav", "mp3", "png", "jpg", "jpeg"]
)

prompt = st.chat_input("Type a message or upload a file...")

def get_model_response(model_choice, messages_or_text, image_url=None):
    if model_choice.startswith("Gemini"):
        return gemini_multimodal(messages_or_text, image_url) if image_url else get_gemini_response(messages_or_text)
    elif model_choice.startswith("OpenAI"):
        return multimodal_response(messages_or_text, image_url) if image_url else get_openai_response(messages_or_text)
    elif model_choice.startswith("Groq"):
        return get_groq_response(messages_or_text)
    return "Model not available."

image_url = None
if uploaded_file is not None:
    if uploaded_file.type.startswith("image"):
        try:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image", use_column_width=True)
            uploaded_image_path = f"temp_{int(time.time())}_{uploaded_file.name}"
            img.save(uploaded_image_path)
            image_url = uploaded_image_path
            st.session_state.uploaded_images[st.session_state.current_session] = uploaded_image_path
            st.info("Image uploaded and saved for visual Q&A. Use `/askimage your question` to query it.")
        except Exception as e:
            st.error(f"Image handling error: {e}")
    elif uploaded_file.type.startswith("audio"):
        with st.spinner("Transcribing audio..."):
            user_text = transcribe_audio(uploaded_file)
        st.session_state.chat_sessions[st.session_state.current_session].append({"role": "user", "content": user_text})
        st.markdown(f'<div class="chat-message user">{user_text}</div>', unsafe_allow_html=True)
        with st.spinner("Generating response..."):
            response = get_model_response(model_choice, user_text)
        st.markdown(f'<div class="chat-message assistant">{response}</div>', unsafe_allow_html=True)
        st.session_state.chat_sessions[st.session_state.current_session].append({"role": "assistant", "content": response})
        with st.spinner("Generating speech..."):
            tts_file = speak_text(response)
        if tts_file and not tts_file.startswith("⚠️"):
            try:
                st.audio(tts_file)
            except Exception:
                st.info(f"TTS generated at: {tts_file}")
        else:
            st.info(tts_file)

if prompt:
    st.session_state.chat_sessions[st.session_state.current_session].append({"role": "user", "content": prompt})
    st.markdown(f'<div class="chat-message user">{prompt}</div>', unsafe_allow_html=True)

    if prompt.lower().startswith("/image"):
        image_prompt = prompt.replace("/image", "").strip()
        if not image_prompt:
            response = "⚠️ Please provide an image prompt after /image"
            st.markdown(f'<div class="chat-message assistant">{response}</div>', unsafe_allow_html=True)
        else:
            with st.spinner("Generating image..."):
                img_url = generate_image(image_prompt)
            if isinstance(img_url, str) and img_url.startswith("⚠️"):
                st.markdown(f'<div class="chat-message assistant">{img_url}</div>', unsafe_allow_html=True)
                response = img_url
            else:
                try:
                    if img_url.startswith("data:image"):
                        header, b64 = img_url.split(",", 1)
                        img_bytes = base64.b64decode(b64)
                        st.image(img_bytes, caption=f"Generated: {image_prompt}", use_column_width=True)
                    else:
                        st.image(img_url, caption=f"Generated: {image_prompt}", use_column_width=True)
                except Exception:
                    st.write("Generated image location:", img_url)
                st.markdown(f'<div class="chat-message assistant">✅ Image generated for: "{image_prompt}"</div>', unsafe_allow_html=True)
                response = f"Image URL: {img_url}"

    elif prompt.lower().startswith("/askimage"):
        question = prompt.replace("/askimage", "").strip()
        uploaded_img = st.session_state.uploaded_images.get(st.session_state.current_session)
        if not uploaded_img:
            response = "⚠️ No image found for this session. Upload an image in the sidebar first."
            st.markdown(f'<div class="chat-message assistant">{response}</div>', unsafe_allow_html=True)
        else:
            with st.spinner("Analyzing image..."):
                response = get_model_response(model_choice, question, uploaded_img)
            st.markdown(f'<div class="chat-message assistant">{response}</div>', unsafe_allow_html=True)

    else:
        with st.spinner("Generating response..."):
            history = st.session_state.chat_sessions[st.session_state.current_session]
            try:
                response = get_model_response(model_choice, history, image_url)
            except Exception as e:
                response = f"⚠️ Error while fetching model response: {e}"
        st.markdown(f'<div class="chat-message assistant">{response}</div>', unsafe_allow_html=True)

    st.session_state.chat_sessions[st.session_state.current_session].append({"role": "assistant", "content": response})

st.download_button(
    label="💾 Download Chat",
    data="\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.chat_sessions[st.session_state.current_session]]),
    file_name=f"{st.session_state.current_session}.txt",
    mime="text/plain"
)