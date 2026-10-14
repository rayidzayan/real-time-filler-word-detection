import tempfile
import os
import io
import wave
import re
from pathlib import Path
from faster_whisper import WhisperModel

FILLER_PATTERNS = {
    "um": r"\bum+\b",
    "uh": r"\buh+\b",
    "erm": r"\berm+\b",
    "er": r"\ber\b",
    "hmm": r"\bhmm+\b",
    "you know": r"\byou\s+know\b",
    "I mean": r"\bi\s+mean\b",
    "I guess": r"\bi\s+guess\b",
    "I think": r"\bi\s+think\b",
    "kind of": r"\bkind\s+of\b",
    "sort of": r"\bsort\s+of\b",
    "like": r"\blike\b",
    "actually": r"\bactually\b",
    "basically": r"\bbasically\b",
    "literally": r"\bliterally\b",
    "well": r"\bwell\b",
    "so": r"\bso\b",
    "right": r"\bright\b",
    "okay": r"\bokay\b",
    "anyway": r"\banyway\b",
    "you see": r"\byou\s+see\b",
    "I suppose": r"\bi\s+suppose\b",
    "the thing is": r"\bthe\s+thing\s+is\b",
    "to be honest": r"\bto\s+be\s+honest\b",
    "at the end of the day": r"\bat the end of the day\b",
    "in a way": r"\bin a way\b",
    "as you know": r"\bas you know\b",
    "what I mean": r"\bwhat i mean\b",
}

# Load the model only when transcription is requested.
_model = None



def transcribe_audio(audio_data):
    global _model

    if _model is None:
        _model = WhisperModel(
            "medium",
            device="cpu",
            compute_type="int8"
        )

    # Streamlit provides recorded or uploaded audio as bytes.
    # Save it temporarily so Faster-Whisper can read it.
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as temp_file:
            temp_file.write(audio_data)
            temp_path = temp_file.name

        segments, info = _model.transcribe(
            temp_path,
            language="en",
            beam_size=5,
            vad_filter=False,
            condition_on_previous_text=False,
            temperature=0.0,
            initial_prompt=(
                "Natural spoken English. Preserve speech "
                "disfluencies and filler words such as um, uh, "
                "erm, hmm, like, you know, and basically."
            )
        )

        # Consume the segments before deleting the temporary file.
        transcript = " ".join(
            segment.text.strip() for segment in segments
        ).strip()

        return transcript

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)



def analyze_text(text):
    words = re.findall(r"\b[\w']+\b", text)
    detected_fillers = []

    for filler, pattern in FILLER_PATTERNS.items():
        matches = re.findall(
            pattern, text, flags=re.IGNORECASE
        )
        detected_fillers.extend([filler] * len(matches))

    total_words = len(words)
    filler_count = len(detected_fillers)

    filler_rate = (
        filler_count / total_words * 100
        if total_words else 0
    )

    return {
        "total_words": total_words,
        "filler_count": filler_count,
        "filler_rate": round(filler_rate, 2),
        "detected_fillers": detected_fillers,
    }
