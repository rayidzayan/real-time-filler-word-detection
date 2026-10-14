
from faster_whisper import WhisperModel

print("Loading Whisper model...")

model = WhisperModel(
    "medium",
    device="cpu",
    compute_type="int8"
)

print("Transcribing audio...")

segments, info = model.transcribe(
        r"C:\Users\rayid\filler word detector\test_audio.wav (2).wav",
    language="en",
    beam_size=5,
    vad_filter=False,
    condition_on_previous_text=False,
    temperature=0.0,
    initial_prompt="This is natural spoken English. Preserve disfluencies and filler words such as um, uh, erm, hmm, like, you know, and basically."
)

for segment in segments:
    print(segment.text)

print("Test completed!")
