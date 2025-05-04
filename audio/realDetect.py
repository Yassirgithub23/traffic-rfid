import sounddevice as sd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import queue
import time
import threading
import librosa

# Load YAMNet model
model = hub.load("yamnet_model")  # Use local model folder
class_names = []
with open("class.csv", "r") as f:
    next(f)
    for line in f:
        parts = line.strip().split(',')
        class_names.append(parts[2])

# Audio settings
SAMPLE_RATE = 16000
BLOCK_SIZE = int(SAMPLE_RATE * 1)  # 1 second chunks
DETECT_EVERY = 3  # Run model every 3 seconds
PAUSE_DURATION = 45  # seconds to wait after siren

audio_buffer = queue.Queue()
siren_detected_time = 0

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    audio_buffer.put(indata.copy())

# Siren detection thread
def detection_loop():
    global siren_detected_time
    audio_clip = []

    while True:
        try:
            data = audio_buffer.get(timeout=1)
            audio_clip.extend(data[:, 0].tolist())

            if len(audio_clip) >= SAMPLE_RATE * DETECT_EVERY:
                if time.time() - siren_detected_time < PAUSE_DURATION:
                    print("🚨 Siren recently detected. Pausing detection...")
                    audio_clip = []  # clear buffer
                    continue

                audio_np = np.array(audio_clip[:SAMPLE_RATE * DETECT_EVERY])
                audio_clip = audio_clip[SAMPLE_RATE * DETECT_EVERY:]  # keep leftover

                # Resample if needed
                if len(audio_np.shape) > 1:
                    audio_np = librosa.to_mono(audio_np.T)
                audio_np = librosa.resample(audio_np, orig_sr=SAMPLE_RATE, target_sr=16000)

                scores, embeddings, spectrogram = model(audio_np)
                predictions = tf.reduce_mean(scores, axis=0)
                top_class = tf.argmax(predictions)
                class_name = class_names[top_class]

                print(f"[INFO] Detected: {class_name}")

                if 'siren' in class_name.lower():
                    print("🚨 Siren detected! Pausing detection for 45 seconds...")
                    siren_detected_time = time.time()

        except queue.Empty:
            continue

# Start the detection thread
threading.Thread(target=detection_loop, daemon=True).start()

# Start recording
with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, callback=audio_callback):
    print("🎧 Listening for sirens in real-time... Press Ctrl+C to stop.")
    while True:
        time.sleep(0.1)  # Keeps main thread alive
