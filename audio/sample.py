import os
os.environ["YOLO_VERBOSE"] = "False"

import time
import cv2
import signal
import lgpio
import MFRC522
import multiprocessing as mp
from ultralytics import YOLO
import queue

# GPIO pin setup
TRAFFIC_LIGHTS = {
    "NORTH": {"RED": 27, "GREEN": 17},
    "EAST":  {"RED": 12, "GREEN": 16},
    "SOUTH": {"RED": 20, "GREEN": 21},
    "WEST":  {"RED": 6,  "GREEN": 13},
}
BUZZER = 18

chip = lgpio.gpiochip_open(0)
for direction in TRAFFIC_LIGHTS.values():
    lgpio.gpio_claim_output(chip, direction["RED"])
    lgpio.gpio_claim_output(chip, direction["GREEN"])
lgpio.gpio_claim_output(chip, BUZZER)

# Shared state
manager = mp.Manager()
shared = manager.dict({
    "emergency_mode": False,
    "pause_prediction": False,
    "siren_detected_time": 0.0,
    "yolo_detected_time": 0.0,
})

def uidToString(uid):
    return "".join(format(i, '02X') for i in uid)

def turn_all_lights_off():
    for pins in TRAFFIC_LIGHTS.values():
        lgpio.gpio_write(chip, pins["RED"], 0)
        lgpio.gpio_write(chip, pins["GREEN"], 0)

def set_traffic_light(active_direction):
    for direction, pins in TRAFFIC_LIGHTS.items():
        if direction == active_direction:
            print(f"{direction} GREEN")
            lgpio.gpio_write(chip, pins["GREEN"], 1)
            lgpio.gpio_write(chip, pins["RED"], 0)
        else:
            lgpio.gpio_write(chip, pins["GREEN"], 0)
            lgpio.gpio_write(chip, pins["RED"], 1)

def buzzer_alert(duration=2):
    print("Buzzer ON")
    lgpio.gpio_write(chip, BUZZER, 1)
    time.sleep(duration)
    lgpio.gpio_write(chip, BUZZER, 0)
    print("Buzzer OFF")

def cleanup():
    print("Cleaning up...")
    turn_all_lights_off()
    lgpio.gpio_write(chip, BUZZER, 0)
    lgpio.gpiochip_close(chip)
    cv2.destroyAllWindows()
    exit(0)

def dual_detection_triggered(shared):
    return abs(shared["siren_detected_time"] - shared["yolo_detected_time"]) <= 45 and shared["siren_detected_time"] != 0 and shared["yolo_detected_time"] != 0

def check_rfid(shared):
    MIFAREReader = MFRC522.MFRC522()
    card_names = {
        "63C0AB1D": {"Name": "AMBULANCE 1", "Number": "TN 99 AB 0000"},
        "AA81E97A": {"Name": "AMBULANCE 2", "Number": "TN 99 AB 0001"},
    }

    while True:
        (status, _) = MIFAREReader.MFRC522_Request(MIFAREReader.PICC_REQIDL)
        if status == MIFAREReader.MI_OK:
            (status, uid) = MIFAREReader.MFRC522_SelectTagSN()
            if status == MIFAREReader.MI_OK:
                uid_str = uidToString(uid)
                print(f"RFID Detected: {uid_str}")
                if uid_str in card_names and not shared["emergency_mode"]:
                    print(f"Authenticated: {card_names[uid_str]['Name']}")
                    shared["emergency_mode"] = True
                    shared["pause_prediction"] = True
                    buzzer_alert()
                    turn_all_lights_off()
                    set_traffic_light("NORTH")
                    time.sleep(15)
                    shared["emergency_mode"] = False
                    shared["pause_prediction"] = False
                    turn_all_lights_off()

def normal_traffic_cycle(shared):
    directions = ["NORTH", "EAST", "SOUTH"]
    current = 0
    while True:
        if not shared["emergency_mode"]:
            set_traffic_light(directions[current])
            time.sleep(10)
            current = (current + 1) % 3

# --- AUDIO PROCESS FUNCTION ---
def audio_process(shared):
    import tensorflow as tf
    import tensorflow_hub as hub
    import sounddevice as sd
    import librosa
    import numpy as np

    SAMPLE_RATE = 16000
    DETECT_EVERY = 3
    audio_clip = []

    yamnet_model = hub.load("yamnet_model")
    class_names = []
    with open("class.csv", "r") as f:
        next(f)
        for line in f:
            parts = line.strip().split(',')
            class_names.append(parts[2])

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(status)
        audio_clip.extend(indata[:, 0].tolist())

    with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, callback=audio_callback):
        while True:
            if len(audio_clip) >= SAMPLE_RATE * DETECT_EVERY:
                audio_np = np.array(audio_clip[:SAMPLE_RATE * DETECT_EVERY])
                audio_clip[:] = audio_clip[SAMPLE_RATE * DETECT_EVERY:]

                try:
                    if len(audio_np.shape) > 1:
                        audio_np = librosa.to_mono(audio_np.T)
                    audio_np = librosa.resample(audio_np, orig_sr=SAMPLE_RATE, target_sr=16000)

                    scores, embeddings, spectrogram = yamnet_model(audio_np)
                    predictions = tf.reduce_mean(scores, axis=0)
                    top_class = tf.argmax(predictions)
                    class_name = class_names[top_class]

                    if 'siren' or 'emergency' in class_name.lower():
                        shared["siren_detected_time"] = time.time()
                        print("Siren Detected!")
                        if dual_detection_triggered(shared):
                            print("Dual detection confirmed (AUDIO + VISION)")
                except Exception as e:
                    print(f"[Audio Error] {e}")

# --- MAIN ---
def run_yolo(shared):
    model = YOLO("best.pt")
    cap = cv2.VideoCapture("http://192.168.214.135:4747/video")
    if not cap.isOpened():
        print("Cannot open IP camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        if not shared["pause_prediction"]:
            results = model.predict(source=frame, imgsz=160, conf=0.8, stream=False, show=False)
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0].item())
                    label = model.names[cls_id].lower()
                    if label == "ambulance":
                        shared["yolo_detected_time"] = time.time()
                        print("YOLO: Ambulance visually detected")
                        shared["pause_prediction"] = True

                        if dual_detection_triggered(shared):
                            print("Dual detection matched! Triggering emergency mode...")
                            shared["emergency_mode"] = True
                            buzzer_alert()
                            turn_all_lights_off()
                            set_traffic_light("NORTH")
                            time.sleep(15)
                            shared["emergency_mode"] = False
                            shared["pause_prediction"] = False
                            turn_all_lights_off()
                        else:
                            print("⚠Waiting for siren confirmation within 45s")
                        break
            for r in results:
                frame = r.plot()

        cv2.imshow("Live YOLO Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    signal.signal(signal.SIGINT, lambda sig, frame: cleanup())

    mp.Process(target=audio_process, args=(shared,), daemon=True).start()
    mp.Process(target=check_rfid, args=(shared,), daemon=True).start()
    mp.Process(target=normal_traffic_cycle, args=(shared,), daemon=True).start()

    run_yolo(shared)
