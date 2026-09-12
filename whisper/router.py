import gc
import io
import logging
import os
import sys
from datetime import datetime
from typing import Sequence, Iterable

import torch
import torchaudio
from fastapi import FastAPI, HTTPException, UploadFile, status
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pydantic import BaseModel

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
WHISPER_MODEL: WhisperModel | None = None

HF_TOKEN = os.environ.get("HF_TOKEN")
DEVICE = os.environ.get("DEVICE", "cpu")


def load_model():
    diarization_checkpoint = os.environ.get("DIARIZATION_MODEL", "pyannote/speaker-diarization-community-1")
    global DIARIZATION_MODEL
    logger.info(f"Loading {diarization_checkpoint} with {HF_TOKEN[:8]=}")
    DIARIZATION_MODEL = Pipeline.from_pretrained(diarization_checkpoint, token=HF_TOKEN)
    DIARIZATION_MODEL.to(torch.device(DEVICE))
    logger.info(f"Loaded {diarization_checkpoint} on {DEVICE}")

    whisper_checkpoint = os.environ.get("WHISPER_MODEL", "Systran/faster-whisper-small")
    global WHISPER_MODEL
    logger.info(f"Loading {whisper_checkpoint} with {HF_TOKEN[:8]=}")
    WHISPER_MODEL = WhisperModel(whisper_checkpoint, DEVICE)
    logger.info(f"Loaded {whisper_checkpoint} on {DEVICE}")


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
    version="12-09-2026",
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


def combine_transcription_and_diarization(transcription: Sequence, diarization: Iterable):

    turns: list[str] = []
    next_sequence_start = 0

    for turn, _ in diarization:
        current_turn = []

        for i in range(next_sequence_start, len(transcription)):
            word = transcription[i]

            word_mid_time = (word.start + word.end) / 2

            logger.warning(f"{word=} {word_mid_time=} {turn.end=} {turn.start=}")

            if word_mid_time <= turn.end and word_mid_time > turn.start:
                logger.error("APPENDING")
                current_turn.append(word.word)
            else:
                logger.error("BREAK")
                break

        logger.info(f"LINE= {current_turn=}")

        turns.append(" ".join(current_turn))
        next_sequence_start = i+1

    return turns


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

    logger.info(f"Start transctibing and diarization on {DEVICE}")
    whisper_waveform = waveform.squeeze().numpy().astype("float32")
    whisper_output, whisper_info = WHISPER_MODEL.transcribe(whisper_waveform, word_timestamps=True, vad_filter=True)
    words = []
    for segment in whisper_output:
        for word in segment.words:
            words.append(word)
    logger.debug(f"Done transcribe: {words=}")

    diarization_output = DIARIZATION_MODEL({"waveform": waveform, "sample_rate": sample_rate})
    diarization = list(diarization_output.speaker_diarization)
    logger.debug(f"Done diarization: {diarization=}")
    logger.info(f"Done transctibing and diarization on {DEVICE}")

    logger.info("Start constructing dialog from models' output")
    lines = combine_transcription_and_diarization(words, diarization)
    logger.info(f"Done constructing dialog from models' output: {lines}")

    turns = []
    for (turn, speaker), line in zip(diarization, lines, strict=True):
        turn = DialogTurnModel(speaker=speaker, turn_start=turn.start, turn_end=turn.end, line=line)
        turns.append(turn)

    logger.info(f"Done constructing dialog from models' output, total turns: {len(turns)}")

    return DialogModel(turns=turns)
