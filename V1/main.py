import sounddevice as sd
import soundfile as sf
import numpy as np
import tempfile
import os
import ollama
import pyttsx3
from faster_whisper import WhisperModel

print("Loading Whisper model...")
whisper = WhisperModel("base", device="cpu", compute_type="int8")


SAMPLE_RATE = 16000
CHUNK_DURATION = 0.1        # read mic every 100ms
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
SILENCE_THRESHOLD = 500     # volume level to detect speech
SILENCE_TIMEOUT = 1.5       # seconds of silence before stopping


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
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": "You are a helpful voice assistant. Keep responses short and conversational."},
            {"role": "user", "content": user_text}
        ]
    )
    return response["message"]["content"]


def speak(text):
    print(f"Speaking: {text}")
    engine = pyttsx3.init()
    engine.setProperty("rate", 175)
    engine.say(text)
    engine.runAndWait()
    engine.stop()


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