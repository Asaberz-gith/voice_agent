# 🎙️ Agentic Voice AI — Built from Scratch

A fully local, free, production-grade voice AI agent built step by step.
100% offline. No paid APIs. No cloud dependencies.

## Stack
- 🎙️ STT Faster-Whisper (local)
- 🧠 LLM Llama 3.2 via Ollama (local)
- 🔊 TTS pyttsx3 (local)

## Versions

 Version  What it does 
------
 V0  Basic pipeline — speak, transcribe, think, respond 
 V1  Real Voice Activity Detection — no button press needed 
 V2  Conversation memory — remembers full session context 

## Setup

### 1. Install Ollama
Download from [ollama.com](httpsollama.com) then
```bash
ollama pull llama3.2
```

### 2. Install Python deps
```bash
pip install faster-whisper pyttsx3 sounddevice soundfile numpy ollama
```

### 3. Run any version
```bash
cd v2
python main.py
```

## Roadmap
- [ ] V3 — Better TTS voice (replace pyttsx3)
- [ ] V4 — Web search tool
- [ ] V5 — Streaming responses
- [ ] V6 — Production deployment