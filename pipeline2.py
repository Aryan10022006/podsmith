import os
import json
import csv
import torch
import whisper
from pydub import AudioSegment
import librosa
import numpy as np
from pyannote.audio import Pipeline
from transformers import pipeline as hf_pipeline, Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from sentence_transformers import SentenceTransformer
import yake
import warnings
import time
from pyannote.core import Annotation, Segment # Ensure these are imported for diarization reconstruction
# Removed AutoFeatureExtractor, AutoModelForAudioClassification as they are not used with SpeechBrain model
from speechbrain.inference import foreign_class
import torchaudio # Added torchaudio for SpeechBrain compatibility

# Suppress specific warnings from libraries if they are not critical to functionality
# Suppress the UserWarning related to Hugging Face Hub symlinks on Windows.
# We make the regex more general to avoid issues with backslashes in paths.
warnings.filterwarnings("ignore", message="`huggingface_hub` cache-system uses symlinks by default")
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")
warnings.filterwarnings("ignore", message="Passing `gradient_checkpointing` to a config initialization is deprecated") # Also saw this in your traceback

# Set your Hugging Face API token as an environment variable
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("HF_TOKEN")
os.environ["HUGGINGFACE_HUB_TOKEN"] = api_key
# Determine the device for PyTorch operations
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Global cache for loaded models to prevent redundant loading
_whisper_model = None
_pyannote_pipeline = None
_sentence_transformer_model = None
_text_emotion_classifier = None
# _ser_processor = None # No longer needed with SpeechBrain's foreign_class
# _ser_model = None # No longer needed with SpeechBrain's foreign_class
_ser_classifier = None # This now holds the SpeechBrain classifier
_yake_extractor = None

# --- Model Loading Functions ---
def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        try:
            print("Loading Whisper model (large-v2) for the first time...")
            _whisper_model = whisper.load_model("large-v2", device=DEVICE)
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model: {e}")
    return _whisper_model

def get_pyannote_pipeline():
    global _pyannote_pipeline
    if _pyannote_pipeline is None:
        if "HUGGINGFACE_HUB_TOKEN" not in os.environ or not os.environ["HUGGINGFACE_HUB_TOKEN"]:
            raise ValueError("HUGGINGFACE_HUB_TOKEN environment variable not set or empty. Please set it to use pyannote.audio.")
        try:
            print("Loading pyannote.audio speaker diarization pipeline (pyannote/speaker-diarization-3.1) for the first time...")
            # Make sure you have accepted the user conditions for pyannote/speaker-diarization-3.1 and pyannote/segmentation-3.0
            _pyannote_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=os.environ["HUGGINGFACE_HUB_TOKEN"])
            _pyannote_pipeline.to(torch.device(DEVICE)) # Move pipeline to device if it supports it
        except Exception as e:
            raise RuntimeError(f"Failed to load pyannote.audio pipeline. Make sure your Hugging Face token is correct and you have accepted the user conditions for pyannote/speaker-diarization-3.1 and pyannote/segmentation-3.0 on Hugging Face: {e}")
    return _pyannote_pipeline

def get_sentence_transformer_model():
    global _sentence_transformer_model
    if _sentence_transformer_model is None:
        try:
            print("Loading SentenceTransformer model (all-MiniLM-L6-v2) for the first time...")
            _sentence_transformer_model = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
        except Exception as e:
            raise RuntimeError(f"Failed to load SentenceTransformer model: {e}")
    return _sentence_transformer_model

def get_text_emotion_classifier():
    global _text_emotion_classifier
    if _text_emotion_classifier is None:
        try:
            print("Loading text emotion recognition model (j-hartmann/emotion-english-distilroberta-base) for the first time...")
            _text_emotion_classifier = hf_pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                return_all_scores=False,
                device=0 if DEVICE == "cuda" else -1
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load text emotion classifier: {e}")
    return _text_emotion_classifier

def get_ser_models():
    global _ser_classifier
    if _ser_classifier is None:
        try:
            print("Loading SpeechBrain Emotion Recognition model (speechbrain/emotion-recognition-wav2vec2-IEMOCAP) for the first time...")
            # This is the SpeechBrain way to load their models
            _ser_classifier = foreign_class(
                source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                pymodule_file="custom_interface.py", # This is often the case for SpeechBrain models on HF
                classname="CustomEncoderWav2vec2Classifier" # Check the model card for the exact classname
            )
            # You might need to move it to device if it doesn't do it automatically
            _ser_classifier.to(DEVICE) # Ensure this method exists for the SpeechBrain object

        except Exception as e:
            raise RuntimeError(f"Failed to load Speech Emotion Recognition models: {e}")
    return _ser_classifier

def get_yake_extractor(max_keywords: int = 5):
    global _yake_extractor
    if _yake_extractor is None:
        print("Initializing YAKE Keyword Extractor...")
        _yake_extractor = yake.KeywordExtractor(top=max_keywords, lan="en", n=1, dedupLim=0.9, dedupFunc='seqm', features=None)
    return _yake_extractor

# --- Intermediate Result Saving/Loading ---
def save_intermediate_json(data, output_dir, filename_base, step_name):
    """Saves intermediate results to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{filename_base}_{step_name}.json")
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Intermediate result for '{step_name}' saved to: {filepath}")
        return filepath
    except Exception as e:
        print(f"Error saving intermediate {step_name} to {filepath}: {e}")
        return None

def load_intermediate_json(output_dir, filename_base, step_name):
    """Loads intermediate results from a JSON file."""
    filepath = os.path.join(output_dir, f"{filename_base}_{step_name}.json")
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"Loaded intermediate result for '{step_name}' from: {filepath}")
            return data
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {filepath} for {step_name}: {e}. File might be corrupted.")
            return None
        except Exception as e:
            print(f"Error loading intermediate {step_name} from {filepath}: {e}")
            return None
    return None

# --- Step 1: Input Audio Preprocessing ---
def preprocess_audio(input_audio_path: str, output_dir: str, filename_base: str, max_duration_sec: int = 300) -> str:
    """
    Standardizes and trims long audio files to a manageable length.

    Args:
        input_audio_path (str): Path to the input audio file.
        output_dir (str): Directory to save the processed WAV file.
        filename_base (str): Base name for the output WAV file.
        max_duration_sec (int): Maximum duration of the output audio in seconds.

    Returns:
        str: Path to the processed WAV file, or None if an error occurred.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_wav_path = os.path.join(output_dir, f"{filename_base}_trimmed.wav")

    if os.path.exists(output_wav_path):
        print(f"Preprocessed audio already exists at {output_wav_path}. Skipping preprocessing.")
        return output_wav_path

    try:
        print(f"Loading audio from {input_audio_path}...")
        audio = AudioSegment.from_file(input_audio_path)
        trimmed_audio = audio[: max_duration_sec * 1000] # pydub uses milliseconds
        trimmed_audio.export(output_wav_path, format="wav")
        print(f"Audio trimmed to {len(trimmed_audio) / 1000:.2f} seconds and saved to {output_wav_path}")
        return output_wav_path
    except FileNotFoundError:
        print(f"Error: Input audio file not found at {input_audio_path}")
        return None
    except Exception as e:
        print(f"Error during audio preprocessing for {input_audio_path}: {e}")
        return None

# --- Step 2: Speech-to-Text Transcription ---
def transcribe_audio(wav_path: str) -> dict:
    """
    Converts spoken audio into accurate text using OpenAI's Whisper model.

    Args:
        wav_path (str): Path to the WAV audio file.

    Returns:
        dict: Whisper transcription result containing 'text' and 'segments'.
    """
    model = get_whisper_model()
    print("Transcribing audio...")
    try:
        result = model.transcribe(wav_path)
        print("Transcription complete.")
        return result
    except Exception as e:
        raise RuntimeError(f"Error during Whisper transcription of {wav_path}: {e}")

# --- Step 3: Speaker Diarization ---
def perform_speaker_diarization(wav_path: str):
    """
    Automatically detects and segments who spoke when in the audio using pyannote.audio.

    Args:
        wav_path (str): Path to the WAV audio file.

    Returns:
        pyannote.core.Annotation: Diarization result.
    """
    pipeline = get_pyannote_pipeline()
    print(f"Performing speaker diarization on {wav_path}...")
    try:
        diarization = pipeline(wav_path)
        print("Diarization complete.")
        return diarization
    except Exception as e:
        raise RuntimeError(f"Error during pyannote speaker diarization of {wav_path}: {e}")

# --- Step 4: Mapping Speakers to Transcript Segments ---
def map_speakers_to_segments(whisper_segments: list, diarization: Annotation) -> list:
    """
    Combines Whisper's text segments with diarization speaker labels.

    Args:
        whisper_segments (list): List of segments from Whisper transcription.
        diarization (pyannote.core.Annotation): Diarization result from pyannote.audio.

    Returns:
        list: Enriched list of segments with speaker information.
    """
    mapped_segments = []
    print("Mapping speakers to transcript segments...")
    for seg in whisper_segments:
        start_whisper = seg['start']
        end_whisper = seg['end']
        
        overlapping_speakers = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            overlap_start = max(start_whisper, turn.start)
            overlap_end = min(end_whisper, turn.end)
            overlap_duration = max(0, overlap_end - overlap_start)

            if overlap_duration > 0:
                overlapping_speakers.append((speaker, overlap_duration))
        
        speaker_id = "Unknown"
        if overlapping_speakers:
            speaker_counts = {}
            for speaker, duration in overlapping_speakers:
                speaker_counts[speaker] = speaker_counts.get(speaker, 0) + duration
            
            speaker_id = max(speaker_counts, key=speaker_counts.get)
        
        mapped_segments.append({
            "start": start_whisper,
            "end": end_whisper,
            "text": seg['text'].strip(),
            "speaker": speaker_id
        })
    print("Speakers mapped to transcript segments complete.")
    return mapped_segments

# --- Step 5: Audio Feature Extraction ---
def extract_audio_features(wav_path: str, segments: list, sr: int = 16000) -> list:
    """
    Extracts relevant audio descriptors (MFCC, chroma, RMS energy) for each speech segment.

    Args:
        wav_path (str): Path to the WAV audio file.
        segments (list): List of audio segments with start and end times.
        sr (int): Sampling rate for audio loading.

    Returns:
        list: List of dictionaries, each containing audio features for a segment.
    """
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"Audio file not found for feature extraction: {wav_path}")

    print(f"Loading audio for feature extraction from {wav_path}...")
    try:
        y, loaded_sr = librosa.load(wav_path, sr=sr)
        if loaded_sr != sr:
            print(f"Warning: Audio loaded with SR {loaded_sr}, but {sr} was requested for feature extraction. Resampling might occur.")
    except Exception as e:
        raise RuntimeError(f"Error loading audio for feature extraction from {wav_path}: {e}")

    audio_features_list = []
    print("Extracting audio features from segments...")
    for i, seg in enumerate(segments):
        start_sample = int(seg['start'] * sr)
        end_sample = int(seg['end'] * sr)
        
        mfccs = np.zeros(13).tolist() # Default 13 MFCCs
        chroma = np.zeros(12).tolist() # Default 12 chroma features
        rms = 0.0

        if start_sample >= len(y) or end_sample <= 0 or start_sample >= end_sample:
            print(f"Warning: Segment {i} out of bounds or zero length ({seg['start']:.2f}-{seg['end']:.2f}s). Assigning zero features.")
        else:
            segment_audio = y[start_sample:min(end_sample, len(y))]
            
            if len(segment_audio) > 0:
                mfccs = librosa.feature.mfcc(y=segment_audio, sr=sr, n_mfcc=13)
                mfccs = np.mean(mfccs.T, axis=0).tolist()

                chroma = librosa.feature.chroma_stft(y=segment_audio, sr=sr)
                chroma = np.mean(chroma.T, axis=0).tolist()

                rms = librosa.feature.rms(y=segment_audio)
                rms = np.mean(rms).item()
            else:
                print(f"Warning: Segment {i} resulted in empty audio after slicing. Assigning zero features.")
        
        audio_features_list.append({
            "mfccs": mfccs,
            "chroma": chroma,
            "rms_energy": rms
        })
    print("Audio feature extraction complete.")
    return audio_features_list

# --- Step 6: Text Embeddings ---
def get_text_embeddings(texts: list) -> list:
    """
    Encodes transcript text segments into numeric vectors using a SentenceTransformer model.

    Args:
        texts (list): List of text strings.

    Returns:
        list: List of text embeddings (each embedding is a list of floats).
    """
    model = get_sentence_transformer_model()
    print("Generating text embeddings...")
    try:
        embeddings = model.encode(texts, convert_to_tensor=False)
        print("Text embeddings generated.")
        return embeddings.tolist()
    except Exception as e:
        raise RuntimeError(f"Error generating text embeddings: {e}")

# --- Step 7a: Text Emotion Recognition ---
def run_text_emotion_analysis(texts: list) -> list:
    """
    Predicts the emotion conveyed in each transcript text segment.

    Args:
        texts (list): List of text strings.

    Returns:
        list: List of emotion labels (e.g., 'joy', 'sadness').
    """
    classifier = get_text_emotion_classifier()
    print("Running text emotion analysis...")
    try:
        results = classifier(texts)
        emotions = [res['label'] for res in results]
        print("Text emotion analysis complete.")
        return emotions
    except Exception as e:
        raise RuntimeError(f"Error during text emotion analysis: {e}")

# --- Step 7b: Speech Emotion Recognition (SER) ---
def run_speech_emotion_recognition(wav_path: str, segments: list, sr: int = 16000) -> list:
    """
    Predicts emotion directly from audio segments.

    Args:
        wav_path (str): Path to the WAV audio file.
        segments (list): List of audio segments with start and end times.
        sr (int): Sampling rate for audio loading.

    Returns:
        list: List of emotion labels from speech.
    """
    # Get the SpeechBrain classifier directly
    classifier = get_ser_models()

    emotions = []
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"Audio file not found for SER: {wav_path}")

    print(f"Loading audio for SER from {wav_path}...")
    try:
        # Use torchaudio.load as it's often more compatible with SpeechBrain
        speech_tensor, loaded_sr = torchaudio.load(wav_path)
        
        # Ensure mono and target sample rate for SpeechBrain
        if speech_tensor.shape[0] > 1: # If stereo, convert to mono
            speech_tensor = torch.mean(speech_tensor, dim=0, keepdim=True)
        
        if loaded_sr != sr:
            print(f"Warning: SER audio loaded with SR {loaded_sr}, but {sr} was requested. Resampling...")
            resampler = torchaudio.transforms.Resample(orig_freq=loaded_sr, new_freq=sr)
            speech_tensor = resampler(speech_tensor)
        
        # Squeeze to remove channel dimension if it's 1, as SpeechBrain often expects (samples,)
        speech_tensor = speech_tensor.squeeze(0) 

    except Exception as e:
        raise RuntimeError(f"Error loading audio for SER from {wav_path}: {e}")

    print("Running speech emotion recognition...")
    for i, seg in enumerate(segments):
        start_sample = int(seg['start'] * sr)
        end_sample = int(seg['end'] * sr)
        
        emotion = "unknown" # Default emotion if segment is invalid or processing fails for it

        if start_sample >= len(speech_tensor) or end_sample <= 0 or start_sample >= end_sample:
            print(f"Warning: SER segment {i} out of bounds or zero length ({seg['start']:.2f}-{seg['end']:.2f}s). Assigning 'unknown' emotion.")
        else:
            # Slice the audio tensor
            segment_audio_tensor = speech_tensor[start_sample:min(end_sample, len(speech_tensor))]
            
            if len(segment_audio_tensor) == 0:
                print(f"Warning: SER segment {i} resulted in empty audio after slicing. Assigning 'unknown' emotion.")
            else:
                try:
                    # SpeechBrain's classify_batch expects a batch dimension (e.g., (1, samples))
                    # and requires it to be on the correct device.
                    # It also expects a float tensor.
                    segment_audio_tensor = segment_audio_tensor.unsqueeze(0).to(DEVICE)
                    
                    # The classifier returns (out_prob, score, index, text_lab)
                    out_prob, score, index, text_lab = classifier.classify_batch(segment_audio_tensor)
                    
                    # text_lab is usually a list containing the predicted label
                    if text_lab and len(text_lab) > 0:
                        emotion = text_lab[0]
                    else:
                        emotion = "unknown_no_label"

                except Exception as e:
                    print(f"Error processing SER for segment {i} ({seg['start']:.2f}-{seg['end']:.2f}s): {e}. Assigning 'unknown' emotion.")
        emotions.append(emotion)
    print("Speech emotion recognition complete.")
    return emotions

# --- Step 8: Keyword Extraction ---
def extract_keywords(texts: list, max_keywords: int = 5) -> list:
    """
    Extracts important keywords or phrases from each transcript segment.

    Args:
        texts (list): List of text strings.
        max_keywords (int): Maximum number of keywords to extract per text.

    Returns:
        list: List of lists, where each inner list contains keywords for a segment.
    """
    kw_extractor = get_yake_extractor(max_keywords)
    all_keywords = []
    print("Extracting keywords...")
    try:
        for text in texts:
            keywords = kw_extractor.extract_keywords(text)
            all_keywords.append([kw for kw, score in keywords])
        print("Keyword extraction complete.")
        return all_keywords
    except Exception as e:
        raise RuntimeError(f"Error during keyword extraction: {e}")

# --- Step 9: Output Storage and Formats ---
def save_final_results(output_dir: str, filename_base: str, full_output_data: dict, segmented_data: list):
    """
    Saves detailed, structured results in JSON, simplified CSV, and raw transcript text.

    Args:
        output_dir (str): Directory to save the output files.
        filename_base (str): Base name for the output files (e.g., "my_audio").
        full_output_data (dict): Dictionary containing the full JSON structure.
        segmented_data (list): List of processed segments for CSV and text output.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save JSON
    json_path = os.path.join(output_dir, f"{filename_base}_final_output.json")
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(full_output_data, f, indent=2, ensure_ascii=False)
        print(f"Final JSON output saved to: {json_path}")
    except Exception as e:
        print(f"Error saving final JSON to {json_path}: {e}")

    # Save CSV
    csv_path = os.path.join(output_dir, f"{filename_base}_summary.csv")
    if segmented_data:
        csv_headers = ["start", "end", "speaker", "text", "text_emotion", "speech_emotion", "keywords", "audio_features_mfccs", "audio_features_chroma", "audio_features_rms_energy", "text_embedding"]
        
        try:
            with open(csv_path, "w", encoding="utf-8", newline='') as f:
                writer = csv.DictWriter(f, fieldnames=csv_headers)
                writer.writeheader()
                for seg in segmented_data:
                    csv_row = seg.copy()
                    
                    audio_features = csv_row.pop("audio_features", {})
                    csv_row["audio_features_mfccs"] = str(audio_features.get("mfccs", []))
                    csv_row["audio_features_chroma"] = str(audio_features.get("chroma", []))
                    csv_row["audio_features_rms_energy"] = audio_features.get("rms_energy", 0.0)

                    if isinstance(csv_row.get("keywords"), list):
                        csv_row["keywords"] = "; ".join(csv_row["keywords"])
                    if isinstance(csv_row.get("text_embedding"), list):
                        csv_row["text_embedding"] = str(csv_row["text_embedding"])
                    
                    writer.writerow(csv_row)
            print(f"Simplified CSV output saved to: {csv_path}")
        except Exception as e:
            print(f"Error saving CSV to {csv_path}: {e}")
    else:
        print("No segmented data to save to CSV.")

    # Save plain text transcript
    txt_path = os.path.join(output_dir, f"{filename_base}_transcript.txt")
    try:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(full_output_data.get("full_transcript", ""))
        print(f"Raw transcript text saved to: {txt_path}")
    except Exception as e:
        print(f"Error saving raw transcript to {txt_path}: {e}")

# --- Main Pipeline Execution ---
def run_full_pipeline(input_audio_path: str, output_dir: str = "output", max_duration_sec: int = 300) -> dict:
    """
    Executes the complete audio analysis pipeline with intermediate result saving.

    Args:
        input_audio_path (str): Path to the input audio file.
        output_dir (str): Directory to save all output files.
        max_duration_sec (int): Maximum duration for trimmed audio in seconds.

    Returns:
        dict: The complete structured output data.
    """
    filename_base = os.path.splitext(os.path.basename(input_audio_path))[0]
    
    print("\n--- Starting Audio Analysis Pipeline ---")

    # Step 1: Input Audio Preprocessing
    print("Step 1: Preprocessing and trimming audio...")
    wav_path = preprocess_audio(input_audio_path, output_dir, filename_base, max_duration_sec)
    if wav_path is None:
        print("Pipeline stopped at Step 1 due to preprocessing error.")
        return {}
    if not os.path.exists(wav_path):
        print(f"Pipeline stopped at Step 1: Preprocessed WAV file not found at {wav_path}.")
        return {}

    # Initialize variables for steps that might be skipped by loading intermediate data
    whisper_result = {"text": "", "segments": []}
    diarization_reconstructed = Annotation()
    diarized_segments = []
    texts = []
    audio_features = []
    text_embeddings = []
    text_emotions = []
    speech_emotions = []
    keywords = []

    # Step 2: Speech-to-Text Transcription
    print("\nStep 2: Performing Speech-to-Text Transcription (Whisper)...")
    try:
        whisper_result_loaded = load_intermediate_json(output_dir, filename_base, "whisper_raw")
        if whisper_result_loaded is None:
            whisper_result = transcribe_audio(wav_path)
            save_intermediate_json(whisper_result, output_dir, filename_base, "whisper_raw")
        else:
            whisper_result = whisper_result_loaded
        whisper_segments = whisper_result["segments"]
        texts = [seg["text"] for seg in whisper_segments] # Populate texts for later use

    except Exception as e:
        print(f"Pipeline stopped at Step 2 due to error: {e}")
        return {}

    # Step 3: Speaker Diarization
    print("\nStep 3: Performing Speaker Diarization (Pyannote)...")
    try:
        diarization_list_loaded = load_intermediate_json(output_dir, filename_base, "diarization_raw")
        if diarization_list_loaded is None:
            diarization_obj = perform_speaker_diarization(wav_path)
            diarization_list = [{"speaker": s, "start": t.start, "end": t.end} for t, _, s in diarization_obj.itertracks(yield_label=True)]
            save_intermediate_json(diarization_list, output_dir, filename_base, "diarization_raw")
        else:
            diarization_list = diarization_list_loaded
        
        # Reconstruct pyannote annotation object from loaded list
        for item in diarization_list:
            diarization_reconstructed[Segment(item['start'], item['end'])] = item['speaker']

    except Exception as e:
        print(f"Pipeline stopped at Step 3 due to error: {e}")
        return {}

    # Step 4: Mapping Speakers to Transcript Segments
    print("\nStep 4: Mapping speakers to transcript segments...")
    try:
        diarized_segments_loaded = load_intermediate_json(output_dir, filename_base, "mapped_segments")
        if diarized_segments_loaded is None:
            diarized_segments = map_speakers_to_segments(whisper_segments, diarization_reconstructed)
            save_intermediate_json(diarized_segments, output_dir, filename_base, "mapped_segments")
        else:
            diarized_segments = diarized_segments_loaded
        texts = [seg["text"] for seg in diarized_segments] # Ensure texts are updated with mapped segments

    except Exception as e:
        print(f"Pipeline stopped at Step 4 due to error: {e}")
        return {}

    if not diarized_segments:
        print("No valid segments found after mapping. Skipping further analysis.")
        return {}

    # Step 5: Audio Feature Extraction
    print("\nStep 5: Extracting audio features (MFCC, Chroma, RMS Energy)...")
    try:
        audio_features_loaded = load_intermediate_json(output_dir, filename_base, "audio_features")
        if audio_features_loaded is None:
            audio_features = extract_audio_features(wav_path, diarized_segments)
            save_intermediate_json(audio_features, output_dir, filename_base, "audio_features")
        else:
            audio_features = audio_features_loaded
    except Exception as e:
        print(f"Pipeline stopped at Step 5 due to error: {e}")
        return {}
        
    # Step 6: Text Embeddings
    print("\nStep 6: Generating text embeddings...")
    try:
        text_embeddings_loaded = load_intermediate_json(output_dir, filename_base, "text_embeddings")
        if text_embeddings_loaded is None:
            text_embeddings = get_text_embeddings(texts)
            save_intermediate_json(text_embeddings, output_dir, filename_base, "text_embeddings")
        else:
            text_embeddings = text_embeddings_loaded
    except Exception as e:
        print(f"Pipeline stopped at Step 6 due to error: {e}")
        return {}

    # Step 7a: Text Emotion Recognition
    print("\nStep 7a: Performing text emotion recognition...")
    try:
        text_emotions_loaded = load_intermediate_json(output_dir, filename_base, "text_emotions")
        if text_emotions_loaded is None:
            text_emotions = run_text_emotion_analysis(texts)
            save_intermediate_json(text_emotions, output_dir, filename_base, "text_emotions")
        else:
            text_emotions = text_emotions_loaded
    except Exception as e:
        print(f"Pipeline stopped at Step 7a due to error: {e}")
        return {}

    # Step 7b: Speech Emotion Recognition
    print("\nStep 7b: Performing speech emotion recognition...")
    try:
        speech_emotions_loaded = load_intermediate_json(output_dir, filename_base, "speech_emotions")
        if speech_emotions_loaded is None:
            speech_emotions = run_speech_emotion_recognition(wav_path, diarized_segments)
            save_intermediate_json(speech_emotions, output_dir, filename_base, "speech_emotions")
        else:
            speech_emotions = speech_emotions_loaded
    except Exception as e:
        print(f"Pipeline stopped at Step 7b due to error: {e}")
        return {}

    # Step 8: Keyword Extraction
    print("\nStep 8: Extracting keywords...")
    try:
        keywords_loaded = load_intermediate_json(output_dir, filename_base, "keywords")
        if keywords_loaded is None:
            keywords = extract_keywords(texts)
            save_intermediate_json(keywords, output_dir, filename_base, "keywords")
        else:
            keywords = keywords_loaded
    except Exception as e:
        print(f"Pipeline stopped at Step 8 due to error: {e}")
        return {}

    # Combine all results into the segments
    final_segments = []
    # Ensure all lists (audio_features, text_embeddings, etc.) have the same length as diarized_segments
    min_len = min(len(diarized_segments), len(audio_features), len(text_embeddings), 
                  len(text_emotions), len(speech_emotions), len(keywords))

    if min_len < len(diarized_segments):
        print(f"Warning: Discrepancy in segment counts. Processing only {min_len} segments out of {len(diarized_segments)}.")

    for i in range(min_len):
        seg_data = diarized_segments[i].copy()
        seg_data["audio_features"] = audio_features[i]
        seg_data["text_embedding"] = text_embeddings[i]
        seg_data["text_emotion"] = text_emotions[i]
        seg_data["speech_emotion"] = speech_emotions[i]
        seg_data["keywords"] = keywords[i]
        final_segments.append(seg_data)

    # Prepare final output structure
    full_output_data = {
        "audio_file": os.path.basename(input_audio_path),
        "full_transcript": whisper_result.get("text", "").strip(),
        "segments": final_segments
    }
    
    simplified_segments = []
    for segment in final_segments:
        simplified_segment = segment.copy()
        simplified_segment.pop("audio_features", None) # Remove audio_features if present
        simplified_segment.pop("text_embedding", None) # Remove text_embedding if present
        simplified_segments.append(simplified_segment)

    simplified_output_data = {
        "audio_file": os.path.basename(input_audio_path),
        "full_transcript": whisper_result.get("text", "").strip(),
        "segments": simplified_segments
    }

    print("\nStep 9: Saving simplified results (no embeddings)...")
    save_final_results(output_dir, f"{filename_base}_no_embeddings", simplified_output_data, simplified_segments)

    print("\n--- Pipeline Complete! ---")
    return full_output_data

    # Step 9: Output Storage and Formats
    print("\nStep 9: Saving final results...")
    save_final_results(output_dir, filename_base, full_output_data, final_segments)

    print("\n--- Pipeline Complete! ---")
    return full_output_data

if __name__ == "__main__":    
    input_audio_file = "test2.mp3"
    output_directory = f"output_data/{input_audio_file}"

    os.makedirs(output_directory, exist_ok=True)

    if not os.path.exists(input_audio_file) or AudioSegment.from_file(input_audio_file).duration_seconds < 10:
        print(f"Creating a dummy '{input_audio_file}' for demonstration purposes.")
        
        sr_dummy = 16000
        duration_dummy = 15 # seconds
        t = np.linspace(0, duration_dummy, int(sr_dummy * duration_dummy), endpoint=False)
        # Combine a sine wave and some random noise
        freq1 = 440 # Hz
        audio_data = 0.5 * np.sin(2 * np.pi * freq1 * t) + 0.1 * np.random.randn(len(t))
        audio_data = audio_data / np.max(np.abs(audio_data)) * 0.5 # Normalize to prevent clipping

        # Convert numpy array to pydub AudioSegment
        audio_segment_dummy = AudioSegment(
            (audio_data * (2**15 - 1)).astype(np.int16).tobytes(), 
            frame_rate=sr_dummy,
            sample_width=2, # 2 bytes = 16-bit
            channels=1
        )
        audio_segment_dummy.export(input_audio_file, format="mp3")
        print(f"Dummy '{input_audio_file}' created ({duration_dummy} seconds). For real analysis, replace it with actual speech.")
    
    # Run the pipeline
    results = run_full_pipeline(input_audio_file, output_dir=output_directory, max_duration_sec=300)

    # You can now inspect the 'results' dictionary or the files in 'output_data'
    print("\n--- Inspection of Final Results (first 2 segments) ---")
    if results and results.get("segments"):
        for i, segment in enumerate(results["segments"][:2]):
            print(f"Segment {i+1}:")
            print(f"  Speaker: {segment.get('speaker', 'N/A')}")
            print(f"  Time: {segment.get('start', 'N/A')}s - {segment.get('end', 'N/A')}s")
            print(f"  Text: {segment.get('text', 'N/A')}")
            print(f"  Text Emotion: {segment.get('text_emotion', 'N/A')}")
            print(f"  Speech Emotion: {segment.get('speech_emotion', 'N/A')}")
            print(f"  Keywords: {', '.join(segment.get('keywords', []))}")
            if 'audio_features' in segment:
                print(f"  MFCCs (first 3): {segment['audio_features'].get('mfccs', [])[:3]}...")
                print(f"  RMS Energy: {segment['audio_features'].get('rms_energy', 'N/A')}")
            else:
                print("  Audio features not available.")
            print(f"  Text Embedding (first 3): {segment.get('text_embedding', [])[:3]}...")
    else:
        print("No segments found or pipeline failed.")