import tempfile
import os
import ollama
import edge_tts
import asyncio
from faster_whisper import WhisperModel
from ddgs import DDGS

print("Loading Whisper model...")
whisper = WhisperModel("base", device="cpu", compute_type="int8")

VOICE = "en-US-JennyNeural"

conversation_history = [
    {"role": "system", "content": """You are a helpful voice assistant. Keep responses short and conversational.
You have a web search tool available. Use it ONLY when the user explicitly asks for:
- Current news or recent events
- Today's weather
- Live sports scores or results
- Current stock prices
- A specific website or company URL
- Any question that explicitly requires up to date information

Do NOT search for:
- Greetings or casual conversation
- General knowledge questions you already know
- Definitions or explanations of common words
- Math or logic questions
- Anything you can answer confidently from your training

To search, your ENTIRE response must be ONLY this, nothing else:
SEARCH: your search query

Otherwise just respond normally without searching."""}
]


def search_web(query):
    print(f"Searching web for: {query}")
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=3):
            results.append(f"{r['title']}: {r['body']} [URL: {r['href']}]")
    return "\n".join(results)


def clean_text(text):
    text = text.replace("°C", " degrees Celsius")
    text = text.replace("°F", " degrees Fahrenheit")
    text = text.replace("°", " degrees")
    text = text.replace("%", " percent")
    text = text.replace("&", " and")
    text = text.replace("$", " dollars")
    text = text.replace("#", " number")
    text = text.replace("*", "")
    text = text.replace(":", "")
    text = text.replace("/", " ")
    text = text.replace("\\", " ")
    text = text.replace("|", "")
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = text.replace("(", "")
    text = text.replace(")", "")
    text = text.replace("[", "")
    text = text.replace("]", "")
    text = text.replace("{", "")
    text = text.replace("}", "")
    if "LINK" in text:
        text = text.split("LINK")[0]
    return text.strip()


def transcribe(audio_bytes):
    tmp = tempfile.NamedTemporaryFile(suffix=".webm", delete=False)
    tmp.write(audio_bytes)
    tmp.close()
    segments, _ = whisper.transcribe(tmp.name, language="en")
    text = " ".join(s.text for s in segments)
    os.unlink(tmp.name)
    return text.strip()


async def synthesize(text):
    text = clean_text(text)
    if not text:
        return None
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp_path = tmp.name
    tmp.close()
    communicate = edge_tts.Communicate(text, VOICE)
    await communicate.save(tmp_path)
    with open(tmp_path, "rb") as f:
        audio_bytes = f.read()
    os.unlink(tmp_path)
    return audio_bytes


async def think_and_speak(user_text, websocket):
    print(f"Thinking about: {user_text}")

    conversation_history.append({"role": "user", "content": user_text})

    response = ollama.chat(
        model="llama3.2",
        messages=conversation_history,
        options={"num_predict": 100}
    )
    reply = response["message"]["content"]

    if "SEARCH:" in reply:
        query = reply.split("SEARCH:")[-1].strip().split("\n")[0].strip()
        search_results = search_web(query)

        conversation_history.append({"role": "assistant", "content": reply})
        conversation_history.append({
            "role": "user",
            "content": f"""Here are the real-time search results:\n{search_results}\n\n
IMPORTANT: Use ONLY these search results to answer.
Do NOT say you lack real-time access.
Do NOT suggest checking other websites.
Summarize the answer in 1-2 sentences and include the most relevant URL at the end in this exact format: LINK: https://..."""
        })

    stream = ollama.chat(
        model="llama3.2",
        messages=conversation_history,
        stream=True,
        options={"num_predict": 100}
    )

    buffer = ""
    full_reply = ""

    for chunk in stream:
        token = chunk["message"]["content"]
        buffer += token
        full_reply += token

        if any(p in buffer for p in [".", "!", "?", ","]):
            current = ""
            for char in buffer:
                current += char
                if char in ".!?,":
                    sentence = current.strip()
                    if sentence and "LINK:" not in sentence:
                        audio = await synthesize(sentence)
                        if audio:
                            await websocket.send_bytes(audio)
                    current = ""
            buffer = current

    # Speak any remaining text
    if buffer.strip() and "LINK:" not in buffer:
        audio = await synthesize(buffer.strip())
        if audio:
            await websocket.send_bytes(audio)

    # Extract and send link if present
    if "LINK:" in full_reply:
        url = full_reply.split("LINK:")[-1].strip().split()[0]
        await websocket.send_text(f"LINK: {url}")

    conversation_history.append({"role": "assistant", "content": full_reply})