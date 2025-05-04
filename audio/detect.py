import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import librosa

# Load the YAMNet model
model = hub.load('yamnet_model')

# Load class labels
import requests
# class_map_path = 'class.csv'
# class_names = requests.get(class_map_path).text.strip().split('\n')[1:]
# class_names = [name.split(',')[2] for name in class_names]
# Load class labels from the local CSV file
class_names = []
with open("class.csv", "r") as f:
    lines = f.readlines()[1:]  # skip the header
    for line in lines:
        parts = line.strip().split(",")
        class_names.append(parts[2])  # 3rd column is display_name


# Load your audio file (must be mono .wav or convert it)
wav_file = 'siren2.wav'
waveform, sr = librosa.load(wav_file, sr=16000)

# Run the model
scores, embeddings, spectrogram = model(waveform)

# Average scores across all frames
mean_scores = tf.reduce_mean(scores, axis=0)

# Get top 5 predictions
top5_indices = tf.argsort(mean_scores, direction='DESCENDING')[:5]

print("\nTop 5 Predicted Sounds:")
for i in top5_indices:
    print(f"- {class_names[i]}: {mean_scores[i].numpy():.3f}")

# Check if siren detected
if any('siren'or 'emergency vehicle' in class_names[i].lower() for i in top5_indices):
    print("\n** Siren detected! **")
else:
    print("\nNo siren detected.")