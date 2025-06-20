import os
import tempfile
import time
import whisper
from pydub import AudioSegment
from pyannote.audio import Pipeline
os.environ["SPEECHBRAIN_LOCAL"] = "True"
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("HF_TOKEN")
# Step 1: Convert MP3 to mono WAV with 16kHz sample rate (max 5 mins)
def trim_to_wav(input_mp3, output_wav, max_sec=300):
    audio = AudioSegment.from_file(input_mp3)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio[:max_sec * 1000].export(output_wav, format="wav")

# Step 2: Transcribe audio with Whisper
def transcribe_whisper(wav_path, model_size="large-v3"):
    model = whisper.load_model(model_size)
    start = time.time()
    result = model.transcribe(wav_path)
    print(f"Whisper transcription time: {time.time() - start:.1f}s")
    return result["segments"]

# Step 3: Perform speaker diarization with pyannote
def diarize_pyannote(wav_path, hf_token):
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
    except Exception as e:
        raise RuntimeError(
            "Could not load pyannote pipeline. Make sure:\n"
            "- You accepted access to https://hf.co/pyannote/speaker-diarization-3.1\n"
            "- You also accepted https://hf.co/pyannote/segmentation-3.0\n"
            f"Original error: {e}"
        )
    diarization = pipeline(wav_path)
    segments = [(turn.start, turn.end, speaker)
                for turn, _, speaker in diarization.itertracks(yield_label=True)]
    return segments

# Step 4: Smarter alignment of Whisper segments to speakers
def align_segments(ts, diarization):
    aligned = []
    for s in ts:
        # Find all overlapping diarization segments
        overlapping = [
            seg for seg in diarization
            if not (s["end"] <= seg[0] or s["start"] >= seg[1])
        ]
        if overlapping:
            # Use the speaker with the longest overlap
            best_seg = max(overlapping, key=lambda seg: min(s["end"], seg[1]) - max(s["start"], seg[0]))
            speaker = f"Speaker_{best_seg[2]}"
        else:
            speaker = "Speaker_unknown"

        aligned.append({
            "start": s["start"],
            "end": s["end"],
            "text": s["text"].strip(),
            "speaker": speaker
        })
    return aligned

# Step 5: Main function
def main(input_mp3):
    hf_token = api_key
    if not hf_token:
        raise RuntimeError(
            "❌ HUGGINGFACE_TOKEN environment variable not found.\n"
            
            
        )

    wav_path = tempfile.mktemp(suffix=".wav")
    trim_to_wav(input_mp3, wav_path)

    print("🔊 Transcribing audio...")
    whisper_segments = transcribe_whisper(wav_path)

    print("🧑‍🤝‍🧑 Performing speaker diarization...")
    diarization_segments = diarize_pyannote(wav_path, hf_token)

    print("📎 Aligning speakers to transcription...")
    aligned_segments = align_segments(whisper_segments, diarization_segments)

    # Step 6: Save results
    with open("raw_transcription.txt", "w", encoding="utf-8") as f:
        for seg in whisper_segments:
            f.write(f"{seg['text'].strip()}\n")

    with open("diarized_transcription.txt", "w", encoding="utf-8") as f:
        for seg in aligned_segments:
            f.write(f"[{seg['start']:.2f}-{seg['end']:.2f}] {seg['speaker']}: {seg['text']}\n")

    print("✅ Done! Output saved to 'raw_transcription.txt' and 'diarized_transcription.txt'.")

if __name__ == "__main__":
    main("test1.mp3")
