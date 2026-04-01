import sounddevice as sd
import soundfile as sf
import numpy as np
import tempfile
import os
import asyncio
import ollama
import edge_tts
from faster_whisper import WhisperModel
from ddgs import DDGS

# Load Whisper model
print("Loading Whisper model...")
whisper = WhisperModel("base", device="cpu", compute_type="int8")

SAMPLE_RATE = 16000
CHUNK_DURATION = 0.1
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
SILENCE_THRESHOLD = 500
SILENCE_TIMEOUT = 1.5

VOICE = "en-US-JennyNeural"  # natural female voice
# Other good options:
# "en-US-GuyNeural"       — male voice
# "en-GB-SoniaNeural"     — British female
# "en-AU-NatashaNeural"   — Australian female

conversation_history = [
    {"role": "system", "content": """You are a helpful voice assistant. Keep responses short and conversational.
You have a web search tool available. You MUST use it for ANY question about:
- Current news, events, or updates
- Weather
- Sports scores or results
- Stock prices
- Anything that may have changed recently

To search, your ENTIRE response must be ONLY this, nothing else:
SEARCH: your search query

Do NOT say you cannot access real-time data. You CAN search. Use it."""}
]


def search_web(query):
    print(f"Searching web for: {query}")
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=3):
            results.append(f"{r['title']}: {r['body']}")
    return "\n".join(results)


def clean_text(text):
    text = text.replace("°C", " degrees Celsius")
    text = text.replace("°F", " degrees Fahrenheit")
    text = text.replace("°", " degrees")
    text = text.replace("%", " percent")
    text = text.replace("&", " and")
    text = text.replace("$", " dollars")
    text = text.replace("#", " number")
    return text.strip()


async def synthesize(text):
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp_path = tmp.name
    tmp.close()
    communicate = edge_tts.Communicate(text, VOICE)
    await communicate.save(tmp_path)
    return tmp_path


def play_audio(path):
    data, sr = sf.read(path)
    sd.play(data, sr)
    sd.wait()
    try:
        os.unlink(path)
    except PermissionError:
        pass


async def speak_sentence(text):
    text = clean_text(text)
    if not text:
        return
    print(f"Speaking: {text}")
    path = await synthesize(text)
    play_audio(path)


def record_audio():
    print("Listening... (speak anytime)")

    recorded = []
    silent_chunks = 0
    speaking = False
    max_silent_chunks = int(SILENCE_TIMEOUT / CHUNK_DURATION)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16") as stream:
        while True:
            chunk, _ = stream.read(CHUNK_SIZE)
            volume = np.abs(chunk).mean()

            if volume > SILENCE_THRESHOLD:
                if not speaking:
                    print("Speech detected...")
                speaking = True
                silent_chunks = 0
                recorded.append(chunk.copy())

            elif speaking:
                recorded.append(chunk.copy())
                silent_chunks += 1
                if silent_chunks >= max_silent_chunks:
                    print("Silence detected, processing...")
                    break

    return np.concatenate(recorded, axis=0)


def save_audio(audio):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, SAMPLE_RATE)
    return tmp.name


def transcribe(filepath):
    print("Transcribing...")
    segments, _ = whisper.transcribe(filepath, language="en")
    text = " ".join(s.text for s in segments)
    return text.strip()


async def think_and_speak(user_text):
    print(f"Thinking about: {user_text}")

    conversation_history.append({"role": "user", "content": user_text})

    # First call — check if search needed
    response = ollama.chat(
        model="llama3.2",
        messages=conversation_history
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
Just summarize the answer directly from the results above in 1-2 sentences."""
        })

        stream = ollama.chat(
            model="llama3.2",
            messages=conversation_history,
            stream=True
        )
    else:
        stream = ollama.chat(
            model="llama3.2",
            messages=conversation_history,
            stream=True
        )

    # Stream tokens and speak sentence by sentence
    buffer = ""
    full_reply = ""

    for chunk in stream:
        token = chunk["message"]["content"]
        buffer += token
        full_reply += token

        if any(p in buffer for p in [".", "!", "?"]):
            current = ""
            spoken = ""
            for char in buffer:
                current += char
                if char in ".!?":
                    await speak_sentence(current.strip())
                    spoken += current
                    current = ""
            buffer = current  # keep incomplete sentence

    # Speak any remaining text
    if buffer.strip():
        await speak_sentence(buffer.strip())

    conversation_history.append({"role": "assistant", "content": full_reply})


async def main():
    print("Voice Agent Ready — speak anytime, Ctrl+C to quit\n")
    while True:
        audio = record_audio()
        filepath = save_audio(audio)

        text = transcribe(filepath)
        os.unlink(filepath)

        if not text:
            print("Didn't catch that, trying again...\n")
            continue

        print(f"You said: {text}")
        await think_and_speak(text)
        print()


if __name__ == "__main__":
    asyncio.run(main())