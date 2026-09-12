from fastapi import FastAPI, File, UploadFile, HTTPException, status
from pyannote.audio import Pipeline
import os
from pydantic import BaseModel
import torch
import torchaudio
import io
import gc
import logging
from datetime import datetime
import sys

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(filename=f'logs/embedding_service_{datetime.now().strftime("%y_%m_%d_%H:%M:%S")}.log'),
        logging.StreamHandler(stream=sys.stdout),
    ],
)

logger = logging.getLogger(__name__)


class DialogModel(BaseModel):
    """
    Dialog contains list of lines
    """

    lines: list[str]


TARGET_SAMPLE_RATE = 16000

model: Pipeline | None = None


def load_model():

    checkpoint = os.environ.get("MODEL", "pyannote/speaker-diarization-community-1")
    hf_token = os.environ.get("HF_TOKEN")
    device = os.environ.get("DEVICE", "cpu")
    global model
    logger.info(f"Loading {checkpoint} with token {hf_token}")
    model = Pipeline.from_pretrained(checkpoint, token=hf_token)
    model.to(torch.device(device))
    logger.info(f"Loaded {checkpoint} on {device}")


def unload_model():
    global model
    logger.info("Unloading model")
    model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Unloaded model")


def lifespan(app: FastAPI):
    load_model()
    yield
    unload_model()


app = FastAPI(title="Shlomo Whisperberg", description="Dialog recognition service", lifespan=lifespan)


@app.post("/transcribe", response_model=DialogModel)
def transcribe(audio_file: UploadFile):
    """
    Audio file -> list of lines
    """
    if not audio_file:
        logger.error("No file!!!")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file!!!")

    file_bytes = audio_file.file.read()

    byte_stream = io.BytesIO(file_bytes)

    try:
        waveform, sample_rate = torchaudio.load(byte_stream)
    except Exception as e:
        logger.error("File load error!!!")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File load error!!!") from e

    # Stereo -> Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 16kHz
    if sample_rate != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sample_rate, TARGET_SAMPLE_RATE)
        waveform = resampler(waveform)
        sample_rate = TARGET_SAMPLE_RATE

    output = model({"waveform": waveform, "sample_rate": sample_rate})

    lines = []
    for turn, speaker in output.speaker_diarization:
        line = f"{speaker} speaks between t={turn.start:.3f}s and t={turn.end:.3f}s"
        lines.append(line)

    logger.info("Success")

    return DialogModel(lines=lines)
