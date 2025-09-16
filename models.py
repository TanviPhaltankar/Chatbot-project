import tempfile
import base64
import json
from openai import OpenAI
import requests
from google import genai

def _save_temp_file(filelike, suffix=""):
    if isinstance(filelike, str):
        return filelike
    data = None
    try:
        data = filelike.read()
    except Exception:
        try:
            data = bytes(filelike)
        except Exception:
            raise RuntimeError("Unsupported file-like object")
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tf.write(data)
    tf.flush()
    tf.close()
    return tf.name

def _get_openai_client():
    api_key = "Your_openai_api_key "
    if not api_key:
        return None, "OPENAI_API_KEY not set."
    try:
        client = OpenAI(api_key=api_key)
        return client, None
    except Exception as e:
        return None, f"OpenAI client init error: {e}"

def get_openai_response(messages):
    client, err = _get_openai_client()
    if not client:
        return f"⚠️ OpenAI unavailable: {err}"
    try:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
            if hasattr(resp, "choices") and resp.choices:
                choice = resp.choices[0]
                if hasattr(choice, "message") and hasattr(choice.message, "content"):
                    return choice.message.content
                if hasattr(choice, "text"):
                    return choice.text
            return str(resp)
        except Exception:
            resp = client.responses.create(
                model="gpt-4.1",
                input=messages
            )
            if hasattr(resp, "output_text"):
                return resp.output_text
            if hasattr(resp, "output") and resp.output:
                texts = []
                for item in resp.output:
                    if isinstance(item, str):
                        texts.append(item)
                    elif isinstance(item, dict) and item.get("content"):
                        texts.append(item["content"])
                if texts:
                    return "\n".join(texts)
            return str(resp)
    except Exception as e:
        return f"⚠️ OpenAI error: {e}"

def generate_image(prompt):
    client, err = _get_openai_client()
    if not client:
        return f"⚠️ OpenAI unavailable: {err}"
    try:
        try:
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                n=1
            )
            if hasattr(response, "data") and response.data:
                first = response.data[0]
                if hasattr(first, "url"):
                    return first.url
                if isinstance(first, dict) and first.get("b64_json"):
                    img_b64 = first["b64_json"]
                    return f"data:image/png;base64,{img_b64}"
            return str(response)
        except Exception:
            resp = client.images.create(
                prompt=prompt,
                n=1,
                size="1024x1024"
            )
            if getattr(resp, "data", None):
                first = resp.data[0]
                if first.get("url"):
                    return first["url"]
                if first.get("b64_json"):
                    return f"data:image/png;base64,{first['b64_json']}"
            return str(resp)
    except Exception as e:
        return f"⚠️ Image generation error: {e}"

def multimodal_response(user_text, image_path):
    client, err = _get_openai_client()
    if not client:
        return f"⚠️ OpenAI unavailable: {err}"
    try:
        if not isinstance(image_path, str):
            image_path = _save_temp_file(image_path, suffix=".png")
        try:
            with open(image_path, "rb") as f:
                img_bytes = f.read()
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            payload = [{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_text},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{b64}"}
                ]
            }]
            resp = client.responses.create(model="gpt-4.1-mini", input=payload)
            if hasattr(resp, "output_text"):
                return resp.output_text
            return str(resp)
        except Exception:
            resp = client.responses.create(
                model="gpt-4.1-mini",
                input=[{"role": "user", "content": [{"type": "input_text", "text": user_text}]}]
            )
            if hasattr(resp, "output_text"):
                return resp.output_text
            return str(resp)
    except Exception as e:
        return f"⚠️ Multi-modal response error: {e}"

def _get_gemini_client():
    key = "Your_gemini_api_key "
    if not genai:
        return None, "Gemini SDK (google.genai) not installed."
    if not key:
        return None, "GEMINI_API_KEY not set."
    try:
        client = genai.Client(api_key=key)
        return client, None
    except Exception as e:
        return None, f"Gemini client init error: {e}"

def get_gemini_response(messages):
    client, err = _get_gemini_client()
    if not client:
        return f"⚠️ Gemini unavailable: {err}"
    try:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        conversation = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=conversation
            )
            if hasattr(response, "text"):
                return response.text
            return str(response)
        except Exception:
            response = client.responses.create(
                model="gemini-2.5-flash",
                input=conversation
            )
            if hasattr(response, "output_text"):
                return response.output_text
            return str(response)
    except Exception as e:
        return f"⚠️ Gemini error: {str(e)}"

def gemini_multimodal(user_text, image_path=None):
    client, err = _get_gemini_client()
    if not client:
        return f"⚠️ Gemini unavailable: {err}"
    try:
        if image_path:
            if not isinstance(image_path, str):
                image_path = _save_temp_file(image_path, suffix=".png")
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
            contents = [
                {"type": "input_text", "text": user_text},
                {"type": "input_image", "image_url": f"data:image/png;base64,{img_b64}"}
            ]
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=contents
                )
                return getattr(response, "text", str(response))
            except Exception:
                response = client.responses.create(
                    model="gemini-2.5-flash",
                    input={"content": contents}
                )
                return getattr(response, "output_text", str(response))
        else:
            return get_gemini_response(user_text)
    except Exception as e:
        return f"⚠️ Gemini multimodal error: {str(e)}"

def get_groq_response(messages):
    api_key = "Your_groq_api_key "
    if not api_key:
        return "⚠️ GROQ_API_KEY not set in environment."
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        if isinstance(messages, str):
            conversation = messages
        else:
            conversation = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        payload = {
            "model": "mixtral-8x7b-32768",
            "messages": [{"role": "user", "content": conversation}],
            "temperature": 0.7
        }
        resp = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=20)
        if resp.status_code != 200:
            return f"⚠️ Groq error: status {resp.status_code} - {resp.text}"
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"⚠️ Groq request error: {e}"

def transcribe_audio(audio_input):
    client, err = _get_openai_client()
    if not client:
        return f"⚠️ OpenAI unavailable: {err}"
    try:
        if not isinstance(audio_input, str):
            audio_path = _save_temp_file(audio_input, suffix=".wav")
        else:
            audio_path = audio_input
        with open(audio_path, "rb") as f:
            try:
                transcription = client.audio.transcriptions.create(model="whisper-1", file=f)
                if hasattr(transcription, "text"):
                    return transcription.text
                if isinstance(transcription, dict) and transcription.get("text"):
                    return transcription["text"]
                return str(transcription)
            except Exception:
                transcription = client.audio.transcribe(model="whisper-1", file=f)
                if hasattr(transcription, "text"):
                    return transcription.text
                return str(transcription)
    except Exception as e:
        return f"⚠️ Transcription error: {e}"

def speak_text(text, voice="verse", instructions="Speak in a neutral tone"):
    client, err = _get_openai_client()
    if not client:
        return f"⚠️ OpenAI unavailable: {err}"
    try:
        try:
            response = client.audio.speech.create(
                model="gpt-4o-mini-tts",
                voice=voice,
                input=text,
                instructions=instructions,
            )
            audio_bytes = None
            if hasattr(response, "audio"):
                audio_bytes = response.audio
            elif hasattr(response, "content"):
                audio_bytes = response.content
            elif isinstance(response, dict) and response.get("audio"):
                audio_bytes = response["audio"]
            if isinstance(audio_bytes, str):
                try:
                    audio_bytes = base64.b64decode(audio_bytes)
                except Exception:
                    audio_url = audio_bytes
                    return f"tts_url:{audio_url}"
            if not audio_bytes:
                return "⚠️ TTS returned no audio bytes."
            audio_path = "tts.mp3"
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)
            return audio_path
        except Exception as e:
            return f"⚠️ TTS error: {e}"
    except Exception as e:
        return f"⚠️ TTS error: {e}"
