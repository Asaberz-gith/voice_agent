import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from agent import transcribe, think_and_speak

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    with open("static/index.html", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")
    current_task = None

    try:
        while True:
            message = await websocket.receive()

            if "text" in message:
                if message["text"] == "INTERRUPT":
                    print("Interrupted by user")
                    if current_task and not current_task.done():
                        current_task.cancel()
                    continue

            if "bytes" in message:
                audio_bytes = message["bytes"]

                if current_task and not current_task.done():
                    current_task.cancel()

                text = transcribe(audio_bytes)
                if not text:
                    continue

                print(f"You said: {text}")
                await websocket.send_text(f"USER: {text}")

                current_task = asyncio.create_task(
                    think_and_speak(text, websocket)
                )
                try:
                    await current_task
                except asyncio.CancelledError:
                    print("Response cancelled")

                await websocket.send_text("DONE")

    except Exception as e:
        print(f"Connection closed: {e}")