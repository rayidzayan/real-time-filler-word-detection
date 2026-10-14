
import io
import wave
import struct
import math

import streamlit as st
from detector import analyze_text, transcribe_audio

st.set_page_config(
    page_title="Filler Word Detection System",
    page_icon="🎙️",
    layout="centered",
)

st.title("🎙️ Real-Time Filler Word Detection")
st.write(
    "Record your voice, review the transcript, "
    "and analyse your speaking performance."
)


def get_audio_metrics(audio_bytes, transcript):
    with wave.open(io.BytesIO(audio_bytes), "rb") as audio:
        sample_rate = audio.getframerate()
        sample_width = audio.getsampwidth()
        frame_count = audio.getnframes()
        channels = audio.getnchannels()
        raw_audio = audio.readframes(frame_count)

    duration = frame_count / sample_rate if sample_rate else 0
    word_count = len(transcript.split())
    wpm = round(word_count / (duration / 60), 1) if duration > 0 else 0

    long_pauses = 0
    total_pause_seconds = 0.0

    if sample_width == 2 and sample_rate > 0:
        bytes_per_frame = sample_width * channels
        window_bytes = int(sample_rate * bytes_per_frame * 0.1)
        window_bytes -= window_bytes % bytes_per_frame

        if window_bytes > 0:
            rms_values = []

            for start in range(0, len(raw_audio), window_bytes):
                chunk = raw_audio[start:start + window_bytes]
                usable = len(chunk) - len(chunk) % 2

                if usable == 0:
                    continue

                samples = struct.unpack(
                    "<" + "h" * (usable // 2), chunk[:usable]
                )
                rms_values.append(
                    math.sqrt(sum(x * x for x in samples) / len(samples))
                )

            if rms_values:
                threshold = max(150, max(rms_values) * 0.06)
                silent_windows = 0

                for rms in rms_values:
                    if rms < threshold:
                        silent_windows += 1
                    else:
                        pause_duration = silent_windows * 0.1
                        if pause_duration >= 0.5:
                            long_pauses += 1
                            total_pause_seconds += pause_duration
                        silent_windows = 0

                pause_duration = silent_windows * 0.1
                if pause_duration >= 0.5:
                    long_pauses += 1
                    total_pause_seconds += pause_duration

    return {
        "duration": round(duration, 2),
        "word_count": word_count,
        "wpm": wpm,
        "long_pauses": long_pauses,
        "pause_seconds": round(total_pause_seconds, 1),
    }


def calculate_performance_score(result, metrics):
    score = 100.0
    score -= min(result["filler_rate"] * 4, 40)
    score -= min(max(metrics["long_pauses"] - 2, 0) * 3, 15)

    wpm = metrics["wpm"]

    if 0 < wpm < 100:
        score -= min((100 - wpm) * 0.25, 15)
    elif wpm > 160:
        score -= min((wpm - 160) * 0.25, 15)

    return round(max(0, min(100, score)))


def get_performance_feedback(result, metrics):
    feedback = []
    wpm = metrics["wpm"]

    if result["filler_rate"] <= 2:
        feedback.append(
            "Good filler-word control. Keep practising naturally."
        )
    else:
        feedback.append(
            "Try replacing unnecessary filler words with "
            "short, deliberate pauses."
        )

    if wpm == 0:
        feedback.append("Speaking speed could not be calculated.")
    elif wpm < 100:
        feedback.append("Try speaking a little more continuously.")
    elif wpm <= 160:
        feedback.append(
            "Your speaking pace is within a useful practice range."
        )
    else:
        feedback.append(
            "Try slowing down so listeners can follow your ideas."
        )

    if metrics["long_pauses"] >= 4:
        feedback.append(
            "Several long pauses were detected. Practise organising "
            "your thoughts before speaking."
        )
    elif metrics["long_pauses"] > 0:
        feedback.append(
            "Some long pauses were detected. Check whether they "
            "were intentional."
        )
    else:
        feedback.append(
            "No long pauses were detected by the audio estimate."
        )

    return feedback


def show_analysis(text, audio_bytes=None):
    result = analyze_text(text)

    st.subheader("Recognized Transcript")
    st.write(text)

    st.subheader("Filler Word Analysis")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Words", result["total_words"])
    col2.metric("Filler Words", result["filler_count"])
    col3.metric("Filler Rate", f'{result["filler_rate"]}%')

    if result["detected_fillers"]:
        st.write(
            "Detected fillers: "
            + ", ".join(result["detected_fillers"])
        )
    else:
        st.info("No supported filler words detected in the transcript.")

    if audio_bytes is not None:
        try:
            metrics = get_audio_metrics(audio_bytes, text)

            st.subheader("Speaking Performance")
            col1, col2, col3 = st.columns(3)
            col1.metric("Speaking Speed", f'{metrics["wpm"]} WPM')
            col2.metric("Long Pauses", metrics["long_pauses"])
            col3.metric("Duration", f'{metrics["duration"]} seconds')

            st.metric(
                "Estimated Time in Long Pauses",
                f'{metrics["pause_seconds"]} seconds',
            )

            score = calculate_performance_score(result, metrics)
            st.progress(score / 100)
            st.write(f"**Practice Performance Score: {score}/100**")
            st.caption(
                "This is an illustrative practice score, "
                "not a validated speaking assessment."
            )

            st.subheader("Personalized Feedback")
            for suggestion in get_performance_feedback(result, metrics):
                st.write("- " + suggestion)

        except (wave.Error, EOFError, ValueError, struct.error) as error:
            st.warning(f"Audio analysis could not be completed: {error}")


def review_transcript(transcript, audio_bytes, key):
    corrected = st.text_area(
        "Review and correct the transcript. Restore any missing um or uh.",
        value=transcript,
        height=150,
        key=key,
    )

    if st.button(
        "Analyse Speaking Performance",
        key=key + "_analyse",
        type="primary",
    ):
        if corrected.strip():
            show_analysis(corrected, audio_bytes)
        else:
            st.warning("Please provide a transcript.")


microphone_tab, upload_tab, text_tab = st.tabs(
    ["🎙️ Microphone", "📁 Upload WAV", "⌨️ Text Analysis"]
)

with microphone_tab:
    recording = st.audio_input("Record your voice")

    if recording is not None:
        st.audio(recording)

        if st.button("Transcribe Recording", key="transcribe_mic"):
            try:
                with st.spinner(
                    "Transcribing locally with Faster-Whisper Medium..."
                ):
                    transcript = transcribe_audio(recording.getvalue())

                st.session_state["mic_transcript"] = transcript
                st.session_state.pop("mic_review_analyse", None)

            except Exception as error:
                st.error(f"Transcription failed: {error}")

        if "mic_transcript" in st.session_state:
            review_transcript(
                st.session_state["mic_transcript"],
                recording.getvalue(),
                "mic_review",
            )

with upload_tab:
    audio_file = st.file_uploader(
        "Choose a WAV audio file",
        type=["wav"],
    )

    if audio_file is not None:
        audio_bytes = audio_file.getvalue()
        st.audio(audio_bytes)

        if st.button(
            "Transcribe Uploaded Audio",
            key="transcribe_upload",
        ):
            try:
                with st.spinner(
                    "Transcribing locally with Faster-Whisper Medium..."
                ):
                    transcript = transcribe_audio(audio_bytes)

                st.session_state["upload_transcript"] = transcript
                st.session_state.pop("upload_review_analyse", None)

            except Exception as error:
                st.error(f"Transcription failed: {error}")

        if "upload_transcript" in st.session_state:
            review_transcript(
                st.session_state["upload_transcript"],
                audio_bytes,
                "upload_review",
            )

with text_tab:
    speech = st.text_area(
        "Enter a transcript",
        placeholder="Um, I think, uh, this project is useful.",
        height=150,
    )

    if st.button("Analyse Text", key="analyse_text"):
        if speech.strip():
            show_analysis(speech)
        else:
            st.warning("Please enter some text.")
