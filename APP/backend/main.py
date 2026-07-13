import base64
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

load_dotenv()

APP_DIR = Path(__file__).resolve().parent
CAMPUS_DATA_PATH = APP_DIR / "campus_data.json"
MAX_AUDIO_BYTES = 25 * 1024 * 1024
MAX_IMAGE_BYTES = 10 * 1024 * 1024

app = FastAPI(title="BELT Campus Guide API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"] ,
    allow_headers=["*"],
)


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or api_key == "replace_with_your_api_key":
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is missing. Put it in backend/.env and restart the server.",
        )
    return OpenAI(api_key=api_key)


def load_campus_data() -> Dict[str, Any]:
    try:
        return json.loads(CAMPUS_DATA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read campus_data.json: {exc}",
        ) from exc


def transcribe_audio(client: OpenAI, upload: UploadFile) -> str:
    audio_bytes = upload.file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="The audio recording was empty.")
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="The recording is larger than 25 MB.")

    original_suffix = Path(upload.filename or "question.m4a").suffix.lower()
    suffix = original_suffix if original_suffix in {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"} else ".m4a"
    temporary_path: Optional[str] = None

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary_file:
            temporary_file.write(audio_bytes)
            temporary_path = temporary_file.name

        with open(temporary_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model=os.getenv("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe"),
                file=audio_file,
                prompt="BELT, UCSC, Silicon Valley Campus, Bowers Avenue, Santa Clara",
            )

        return transcription.text.strip()
    finally:
        if temporary_path:
            try:
                os.remove(temporary_path)
            except OSError:
                pass


def image_to_data_url(upload: UploadFile) -> str:
    image_bytes = upload.file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The camera image was empty.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="The image is larger than 10 MB.")

    content_type = upload.content_type or "image/jpeg"
    if content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Use a JPEG, PNG, or WebP image.")

    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{content_type};base64,{encoded}"


def build_instructions(campus_data: Dict[str, Any]) -> str:
    return f"""
You are BELT, a friendly AI receptionist and indoor campus guide for the UCSC Silicon Valley Campus at 3175 Bowers Avenue, Santa Clara, California.

Your job:
- Answer visitor questions clearly and briefly.
- Use the camera image to read visible signs, recognize ordinary objects, and describe the immediate surroundings.
- Give short, step-by-step directions only when they are supported by readable signs or by the approved campus data below.
- Help with rooms, workshops, restrooms, accessibility routes, reception, and general visitor guidance.

Safety and accuracy rules:
- A generic hallway image does not prove the user's exact location. Never pretend it does.
- If the destination or starting point is uncertain, say what you can see and ask the user to point the camera at a room number, directory, elevator sign, or recognizable landmark.
- Never invent room numbers, floor plans, events, opening hours, or routes.
- If a route is not in the approved data, say that it has not been verified and suggest asking reception.
- Do not identify people, guess identities, or infer sensitive traits from faces or appearance.
- For emergencies, tell the user to contact on-site staff or emergency services instead of trying to navigate solely through the app.
- Keep spoken answers easy to follow: normally 2 to 6 sentences.

Approved campus data:
{json.dumps(campus_data, indent=2)}
""".strip()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "app": "BELT Campus Guide API"}


@app.post("/ask")
def ask_belt(
    audio: Optional[UploadFile] = File(default=None),
    image: Optional[UploadFile] = File(default=None),
    question_text: Optional[str] = Form(default=None),
) -> Dict[str, Any]:
    if audio is None and image is None and not (question_text or "").strip():
        raise HTTPException(status_code=400, detail="Send a voice recording, a typed question, or a camera image.")

    client = get_client()
    campus_data = load_campus_data()

    transcript = transcribe_audio(client, audio) if audio is not None else ""
    typed_question = (question_text or "").strip()
    question = typed_question or transcript

    if not question and image is not None:
        question = "Describe what is visible and explain how BELT can help from here."
    if not question:
        raise HTTPException(status_code=400, detail="BELT could not detect a question. Please try again.")

    content = [{"type": "input_text", "text": question}]
    used_camera = image is not None
    if image is not None:
        content.append(
            {
                "type": "input_image",
                "image_url": image_to_data_url(image),
                "detail": "low",
            }
        )

    try:
        response = client.responses.create(
            model=os.getenv("OPENAI_VISION_MODEL", "gpt-5.6-luna"),
            instructions=build_instructions(campus_data),
            input=[{"role": "user", "content": content}],
            max_output_tokens=350,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI request failed: {exc}") from exc

    answer = response.output_text.strip()
    if not answer:
        raise HTTPException(status_code=502, detail="BELT received an empty AI response.")

    return {
        "transcript": transcript or typed_question,
        "answer": answer,
        "used_camera": used_camera,
    }
