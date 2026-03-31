import sounddevice as sd
import soundfile as sf
import numpy as np
import tempfile
import os
import ollama
from faster_whisper import WhisperModel
from TTS.api import TTS as CoquiTTS
from ddgs import DDGS

# Load Whisper model
print("Loading Whisper model...")
whisper = WhisperModel("base", device="cpu", compute_type="int8")

# Load TTS model
print("Loading TTS model...")
tts_engine = CoquiTTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)

SAMPLE_RATE = 16000
CHUNK_DURATION = 0.1
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
SILENCE_THRESHOLD = 500
SILENCE_TIMEOUT = 1.5

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
    print(f"Raw results: {results}")
    return "\n".join(results)


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


def think(user_text):
    print(f"Thinking about: {user_text}")

    conversation_history.append({"role": "user", "content": user_text})

    response = ollama.chat(
        model="llama3.2",
        messages=conversation_history
    )

    reply = response["message"]["content"]

    # Flexible SEARCH detection
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

        final_response = ollama.chat(
            model="llama3.2",
            messages=conversation_history
        )
        reply = final_response["message"]["content"]

    conversation_history.append({"role": "assistant", "content": reply})
    return reply


def speak(text):
    print(f"Speaking: {text}")
    # Clean special characters Coqui can't pronounce
    text = text.replace("°C", " degrees Celsius")
    text = text.replace("°F", " degrees Fahrenheit")
    text = text.replace("°", " degrees")
    text = text.replace("%", " percent")
    text = text.replace("&", " and")
    text = text.replace("$", " dollars")
    text = text.replace("#", " number")
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()
    tts_engine.tts_to_file(text=text, file_path=tmp_path)
    data, sr = sf.read(tmp_path)
    sd.play(data, sr)
    sd.wait()
    try:
        os.unlink(tmp_path)
    except PermissionError:
        pass


def main():
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
        response = think(text)
        speak(response)
        print()


if __name__ == "__main__":
    main()