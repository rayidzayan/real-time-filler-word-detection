
import os
import re
import tempfile
from collections import Counter

from faster_whisper import WhisperModel


# Only count relatively reliable filler expressions by default.
# Avoid counting ordinary words such as "so", "right", and "actually".
FILLER_PATTERNS = {
    "um": r"\bumm*\b",
    "uh": r"\buh+\b",
    "erm": r"\berm+\b",
    "er": r"\ber+\b",
    "ah": r"\bah+\b",
    "hmm": r"\bhmm+\b",
    "mm": r"\bmm+\b",
    "you know": r"\byou\s+know\b",
    "I mean": r"\bi\s+mean\b",
    "kind of": r"\bkind\s+of\b",
    "sort of": r"\bsort\s+of\b",
}

_model = None


def get_model():
    global _model

    if _model is None:
        _model = WhisperModel(
            "medium",
            device="cpu",
            compute_type="int8"
        )

    return _model


def transcribe_audio(audio_data):
    """
    Convert microphone WAV bytes into text using Faster-Whisper.
    """

    model = get_model()
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as temp_file:

            temp_file.write(audio_data)
            temp_path = temp_file.name

        segments, info = model.transcribe(
            temp_path,
            language="en",

            # Improve recognition quality
            beam_size=8,

            # Keep short hesitation sounds available for recognition
            vad_filter=False,

            # Prevent previous sentence context from influencing
            # the next transcription
            condition_on_previous_text=False,

            # Avoid unnecessary random transcription variation
            temperature=0.0,

            # Help Whisper retain spoken disfluencies
            initial_prompt=(
                "This is a natural English conversation or speech. "
                "Transcribe exactly what the speaker says. "
                "Do not remove repetitions, hesitations, or filler words. "
                "Preserve words and sounds such as um, umm, uh, uhh, "
                "erm, er, ah, hmm, mm, you know, and I mean. "
                "Do not summarize or correct the speaker's grammar."
            )
        )

        transcript_parts = []

        for segment in segments:
            if segment.text:
                transcript_parts.append(segment.text.strip())

        return " ".join(transcript_parts).strip()

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def analyze_text(text):
    """
    Detect filler expressions in a transcript.
    """

    if not text or not text.strip():
        return {
            "total_words": 0,
            "filler_count": 0,
            "filler_rate": 0.0,
            "detected_fillers": [],
        }

    # Count spoken tokens
    words = re.findall(
        r"\b[\w']+\b",
        text.lower()
    )

    detected_fillers = []

    for filler, pattern in FILLER_PATTERNS.items():

        matches = re.findall(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        detected_fillers.extend(
            [filler] * len(matches)
        )

    total_words = len(words)
    filler_count = len(detected_fillers)

    filler_rate = (
        (filler_count / total_words) * 100
        if total_words > 0 else 0
    )

    return {
        "total_words": total_words,
        "filler_count": filler_count,
        "filler_rate": round(filler_rate, 2),
        "detected_fillers": detected_fillers,
    }
