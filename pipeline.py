import os
import json
import torch
import whisper
from pydub import AudioSegment
import librosa
import numpy as np
from pyannote.audio import Pipeline
from transformers import pipeline as hf_pipeline
from sentence_transformers import SentenceTransformer
import yake
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("HF_TOKEN")
os.environ["HUGGINGFACE_HUB_TOKEN"] = api_key
# --- Step 1: Convert & trim audio ---
def convert_and_trim(input_audio, output_wav, max_duration_sec=300):
    audio = AudioSegment.from_file(input_audio)
    trimmed = audio[: max_duration_sec * 1000]
    trimmed.export(output_wav, format="wav")

# --- Step 2: Run Whisper ASR ---
def run_whisper(wav_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("large-v2", device=device)
    result = model.transcribe(wav_path)
    return result

# --- Step 3: Pyannote diarization (auto speaker count) ---
def run_pyannote_diarization(wav_path):
    # You must have your huggingface token set as environment variable
    # export HUGGINGFACE_HUB_TOKEN="your_token_here"
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=os.environ["HUGGINGFACE_HUB_TOKEN"])
    diarization = pipeline(wav_path)
    return diarization

# --- Step 4: Map speakers to Whisper segments ---
def map_speakers_to_segments(whisper_segments, diarization):
    mapped_segments = []
    for seg in whisper_segments:
        start = seg['start']
        end = seg['end']
        # Find overlapping speakers
        overlapping = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if start < turn.end and end > turn.start:
                overlapping.append(speaker)
        if overlapping:
            # Choose most frequent overlapping speaker
            speaker = max(set(overlapping), key=overlapping.count)
        else:
            speaker = "Unknown"
        mapped_segments.append({
            "start": start,
            "end": end,
            "text": seg['text'],
            "speaker": speaker
        })
    return mapped_segments

# --- Step 5: Text embeddings ---
def get_text_embeddings(texts):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, convert_to_tensor=True)
    return embeddings.cpu().numpy()

# --- Step 6: Text Emotion Recognition ---
def run_text_emotion_analysis(texts):
    classifier = hf_pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=False)
    return [classifier(text)[0]['label'] for text in texts]

# --- Step 7: Speech Emotion Recognition ---
def run_speech_emotion_recognition(wav_path, segments):
    from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
    processor = Wav2Vec2Processor.from_pretrained("anton-l/speech_emotion_recognition_wav2vec2_large")
    model = Wav2Vec2ForSequenceClassification.from_pretrained("anton-l/speech_emotion_recognition_wav2vec2_large")

    emotions = []
    speech, sr = librosa.load(wav_path, sr=16000)

    for seg in segments:
        start_sample = int(seg['start'] * sr)
        end_sample = int(seg['end'] * sr)
        segment_audio = speech[start_sample:end_sample]
        inputs = processor(segment_audio, sampling_rate=sr, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_id = logits.argmax(dim=-1).item()
        emotion = model.config.id2label[predicted_id]
        emotions.append(emotion)

    return emotions

# --- Step 8: Keyword extraction ---
def extract_keywords(texts, max_keywords=5):
    kw_extractor = yake.KeywordExtractor(top=max_keywords)
    all_keywords = []
    for text in texts:
        keywords = kw_extractor.extract_keywords(text)
        all_keywords.append([kw for kw, score in keywords])
    return all_keywords

# --- Step 9: Main pipeline ---
def full_pipeline(input_audio, output_dir="output", max_duration=300):
    os.makedirs(output_dir, exist_ok=True)
    wav_path = os.path.join(output_dir, "trimmed.wav")
    print("Step 1: Convert and trim audio...")
    convert_and_trim(input_audio, wav_path, max_duration)

    print("Step 2: Run Whisper ASR...")
    whisper_result = run_whisper(wav_path)
    whisper_segments = whisper_result["segments"]  # contains start, end, text

    print("Step 3: Run Pyannote diarization...")
    diarization = run_pyannote_diarization(wav_path)

    print("Step 4: Map speakers to transcript segments...")
    diarized_segments = map_speakers_to_segments(whisper_segments, diarization)

    texts = [seg["text"] for seg in diarized_segments]

    print("Step 5: Run text emotion analysis...")
    text_emotions = run_text_emotion_analysis(texts)

    print("Step 6: Run speech emotion recognition...")
    speech_emotions = run_speech_emotion_recognition(wav_path, diarized_segments)

    print("Step 7: Extract keywords...")
    keywords = extract_keywords(texts)

    # Add emotions and keywords
    for i, seg in enumerate(diarized_segments):
        seg["text_emotion"] = text_emotions[i]
        seg["speech_emotion"] = speech_emotions[i]
        seg["keywords"] = keywords[i]

    output_json = {
        "audio_file": os.path.basename(input_audio),
        "full_transcript": whisper_result["text"],
        "segments": diarized_segments
    }

    json_path = os.path.join(output_dir, "final_output.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False)

    print(f"Pipeline complete. Results saved to: {json_path}")
    return output_json

if __name__ == "__main__":
    input_audio_file = "test1.mp3"  # your audio file here
    results = full_pipeline(input_audio_file, output_dir="output_data", max_duration=300)
