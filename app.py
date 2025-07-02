# app.py
from fastapi import FastAPI
from transformers import pipeline
import torch
from typing import Any, Dict, Optional
from pydantic import BaseModel
import yaml
from pathlib import Path

app = FastAPI()

# Load the Whisper model
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",
    torch_dtype=torch.float16,  # Use float16 for potentially faster inference on compatible hardware
    device="cuda:0" if torch.cuda.is_available() else "cpu",
)

PARAMS_FILE = Path("generation_params.yaml")


class GenerationParameters(BaseModel):
    """Configurable generation parameters for Whisper."""

    max_new_tokens: Optional[int] = 448
    num_beams: Optional[int] = 1
    condition_on_prev_tokens: Optional[bool] = False
    compression_ratio_threshold: Optional[float] = 1.35
    temperature: Optional[tuple[float, ...]] = (
        0.0,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
    )
    logprob_threshold: Optional[float] = -1.0
    no_speech_threshold: Optional[float] = 0.6
    return_timestamps: Optional[bool] = True
    task: Optional[str] = "transcribe"


def load_parameters() -> Dict[str, Any]:
    """Load parameters from the YAML file, creating it if necessary."""
    if PARAMS_FILE.exists():
        with PARAMS_FILE.open("r") as f:
            data = yaml.safe_load(f) or {}
    else:
        params = GenerationParameters().dict(exclude_none=True)
        with PARAMS_FILE.open("w") as f:
            yaml.safe_dump(params, f)
        data = params
    return data


def save_parameters(params: Dict[str, Any]) -> None:
    """Write parameters to the YAML file."""
    with PARAMS_FILE.open("w") as f:
        yaml.safe_dump(params, f)


@app.get("/v1/model/parameters/")
async def get_parameters() -> Dict[str, Any]:
    """Return the current generation parameters."""
    return load_parameters()


@app.patch("/v1/model/parameters/")
async def update_parameters(params: GenerationParameters):
    """Update generation parameters for the Whisper model."""
    stored = load_parameters()
    stored.update(params.dict(exclude_unset=True))
    save_parameters(stored)
    return {"message": "Parameters updated", "parameters": stored}


@app.post("/v1/audio/transcriptions/")
async def transcribe_audio(audio_path: str):
    """Transcribe an audio file using the Whisper model."""
    try:
        generation_params = load_parameters()
        result = pipe(audio_path, generate_kwargs=generation_params)
        return {"transcription": result["text"]}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8917)
