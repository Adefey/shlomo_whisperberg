import gc
import io
import logging
import os
import sys
from datetime import datetime
from typing import Any

import torch
import torchaudio
from fastapi import FastAPI, HTTPException, UploadFile, status
from pyannote.audio import Pipeline
from pydantic import BaseModel

import whisper

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(filename=f'logs/embedding_service_{datetime.now().strftime("%y_%m_%d_%H-%M-%S")}.log'),
        logging.StreamHandler(stream=sys.stdout),
    ],
)

logger = logging.getLogger(__name__)


class DialogTurnModel(BaseModel):
    speaker: str
    turn_start: float
    turn_end: float
    line: str = ""


class DialogModel(BaseModel):
    """
    Dialog contains list of lines
    """

    turns: list[DialogTurnModel]


TARGET_SAMPLE_RATE = 16000

DIARIZATION_MODEL: Pipeline | None = None
WHISPER_MODEL: Any | None = None

HF_TOKEN = os.environ.get("HF_TOKEN")

WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
DIARIZATION_DEVICE = os.environ.get("DIARIZATION_DEVICE", "cpu")


def load_model():
    whisper_checkpoint = os.environ.get("WHISPER_MODEL", "openai/whisper-small")
    global WHISPER_MODEL
    logger.info(f"Loading {whisper_checkpoint}")
    WHISPER_MODEL = whisper.load_model(whisper_checkpoint, WHISPER_DEVICE)
    DIARIZATION_MODEL.segmentation_batch_size = 8
    DIARIZATION_MODEL.embedding_batch_size = 8
    logger.info(f"Loaded {whisper_checkpoint} on {WHISPER_DEVICE}")

    diarization_checkpoint = os.environ.get("DIARIZATION_MODEL", "pyannote/speaker-diarization-community-1")
    global DIARIZATION_MODEL
    logger.info(f"Loading {diarization_checkpoint}")
    DIARIZATION_MODEL = Pipeline.from_pretrained(diarization_checkpoint, use_auth_token=HF_TOKEN)
    DIARIZATION_MODEL.to(torch.device(DIARIZATION_DEVICE))
    logger.info(f"Loaded {diarization_checkpoint} on {DIARIZATION_DEVICE}")


def unload_model():
    global DIARIZATION_MODEL
    global WHISPER_MODEL
    logger.info("Unloading models")
    DIARIZATION_MODEL = None
    WHISPER_MODEL = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Unloaded models")


def lifespan(app: FastAPI):
    load_model()
    yield
    unload_model()


app = FastAPI(
    title="Shlomo Whisperberg",
    description="Dialog recognition service",
    lifespan=lifespan,
    version="13-09-2026",
)


def preprocess_audio(file_bytes: bytes) -> tuple[torch.Tensor, int]:
    byte_stream = io.BytesIO(file_bytes)
    waveform, sample_rate = torchaudio.load(byte_stream)

    # Stereo -> Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 16kHz
    if sample_rate != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE)
        waveform = resampler(waveform)
        sample_rate = TARGET_SAMPLE_RATE

    return waveform, sample_rate


def combine_transcription_and_diarization(transcription: list, diarization: list):

    if not diarization:
        return []

    # diarization object contains immutable objects, need to convert to list of lists
    diarization = [[turn.start, turn.end] for turn, _ in diarization]
    # precalc - allocate full timeline with intervals
    # Guarantee start from 0 even if actual line does not start from 0
    diarization[0][0] = 0.0
    for i in range(len(diarization)):
        if i > 0:
            diarization[i][0] = diarization[i - 1][1]
        if i < len(diarization) - 1:
            diarization[i][1] = (diarization[i][1] + diarization[i + 1][0]) / 2

    # Guarantee last turn contains all the words
    diarization[-1][1] = float("inf")

    turns = [[] for _ in range(len(diarization))]

    for word in transcription:
        word_mid_time = (word["start"] + word["end"]) / 2
        for i in range(len(diarization)):
            if word_mid_time >= diarization[i][0] and word_mid_time < diarization[i][1]:
                turns[i].append(word["word"])
                break

    turns = [" ".join(turn) for turn in turns]

    return turns


def normalize_line(line: str) -> str:
    normalized_line = " ".join(line.split())
    return normalized_line


@app.post("/transcribe", response_model=DialogModel)
def transcribe(audio_file: UploadFile):
    """
    Audio file -> list of lines
    """
    if not audio_file:
        logger.error("No file!")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file!")

    logger.info("Start loading/processing audio")
    try:
        waveform, sample_rate = preprocess_audio(audio_file.file.read())
    except Exception as e:
        logger.error("File processing error!")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File processing error!") from e
    logger.info("Done loading/processing audio")

    # Call models one by one to reduce peak memory/compute usage

    logger.info(f"Start transctibing and diarization on {WHISPER_DEVICE} + {DIARIZATION_DEVICE}")
    whisper_waveform = waveform.squeeze().numpy().astype("float32")
    whisper_output = WHISPER_MODEL.transcribe(whisper_waveform, word_timestamps=True)

    words = []
    for segment in whisper_output["segments"]:
        if "words" in segment:  # проверка, что слова есть
            for word in segment["words"]:
                words.append(word)
    logger.debug(f"Done transcribe: {words=} {whisper_output['language']=}")

    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Transcription is ready. Diarization in progress...")

    diarization_output = DIARIZATION_MODEL({"waveform": waveform, "sample_rate": sample_rate})
    diarization = [(segment, label) for segment, _, label in diarization_output.itertracks(yield_label=True)]
    logger.debug(f"Done diarization: {diarization=}")

    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"Done transctibing and diarization on {WHISPER_DEVICE} + {DIARIZATION_DEVICE}")

    logger.info("Start constructing dialog from models' output")
    lines = combine_transcription_and_diarization(words, diarization)
    logger.info(f"Done constructing dialog from models' output")

    turns = []
    for (turn, speaker), line in zip(diarization, lines, strict=True):
        line = normalize_line(line)
        if line:
            turn = DialogTurnModel(speaker=speaker, turn_start=turn.start, turn_end=turn.end, line=line)
            turns.append(turn)

    logger.info(f"Done constructing dialog from models' output, total turns: {len(turns)}")

    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return DialogModel(turns=turns)
