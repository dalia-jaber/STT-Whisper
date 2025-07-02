# app.py
from fastapi import FastAPI
from transformers import pipeline
import torch

app = FastAPI()

# Load the Whisper model
# You might want to specify a device if you have a GPU, e.g., device=0 for GPU
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",
    torch_dtype=torch.float16, # Use float16 for potentially faster inference on compatible hardware
    device="cuda:0" if torch.cuda.is_available() else "cpu"
)

@app.post("/v1/audio/transcriptions/")
async def transcribe_audio(audio_path: str):
    """
    Transcribes an audio file using the Whisper model.
    """
    # Replace with actual audio handling (e.g., receiving audio as bytes or file upload)
    # For simplicity, this example assumes a path to a local audio file within the container
    try:
        result = pipe(audio_path, generate_kwargs={"task": "transcribe"})
        return {"transcription": result["text"]}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8917)
