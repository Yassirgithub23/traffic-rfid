import os
os.environ["YOLO_VERBOSE"] = "False"

import cv2
import time
import signal
import lgpio
import MFRC522
import csv
from datetime import datetime
import queue
import numpy as np
import multiprocessing as mp
import threading
from ultralytics import YOLO

LOG_FILE = "detection_logs.csv"
if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE)
with open(LOG_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["time", "method", "ambulance", "ambulance_number"])

def log_detection(source, name="UNKNOWN", number="UNKNOWN"):
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), source, name, number])

# --- GPIO Setup ---
TRAFFIC_LIGHTS = {
    "NORTH": {"RED": 27, "GREEN": 17},
    "EAST":  {"RED": 12, "GREEN": 16},
    "SOUTH": {"RED": 20, "GREEN": 21},
    "WEST":  {"RED": 6,  "GREEN": 13},
}
BUZZER = 18
chip = lgpio.gpiochip_open(0)
for pins in TRAFFIC_LIGHTS.values():
    lgpio.gpio_claim_output(chip, pins["RED"])
    lgpio.gpio_claim_output(chip, pins["GREEN"])
lgpio.gpio_claim_output(chip, BUZZER)

# --- Shared State ---
emergency_mode = mp.Value('b', False)
pause_prediction = mp.Value('b', False)
yolo_detected_time = mp.Value('d', 0.0)
siren_detected_time = mp.Value('d', 0.0)
detection_trigger = mp.Event()

# --- RFID ---
MIFAREReader = MFRC522.MFRC522()
card_names = {
    "1DABC063": {"Name": "AMBULANCE 1", "Number": "TN 99 AB 0000"},
    "AA81E97A": {"Name": "AMBULANCE 2", "Number": "TN 99 AB 0001"},
    "63C0AB1D": {"Name": "AMBULANCE 2", "Number": "TN 99 AB 0002"},
    
}

# --- Helper Functions ---
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

def buzzer_alert(duration=1.5):
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
    cap.release()
    cv2.destroyAllWindows()
    exit(0)

signal.signal(signal.SIGINT, lambda sig, frame: cleanup())

def dual_detection_triggered():
    return abs(siren_detected_time.value - yolo_detected_time.value) <= 45 and \
           siren_detected_time.value != 0 and yolo_detected_time.value != 0

# --- RFID Thread ---
def check_rfid():
    while True:
        (status, _) = MIFAREReader.MFRC522_Request(MIFAREReader.PICC_REQIDL)
        if status == MIFAREReader.MI_OK:
            (status, uid) = MIFAREReader.MFRC522_SelectTagSN()
            if status == MIFAREReader.MI_OK:
                uid_str = uidToString(uid)
                print(f"RFID Detected: {uid_str}")
                if uid_str in card_names and not emergency_mode.value:
                    info = card_names[uid_str]
                    print(f"Authenticated: {info['Name']}  {info['Number']}")
                    log_detection("RFID", info["Name"], info["Number"])
                    emergency_mode.value = True
                    pause_prediction.value = True
                    buzzer_alert()
                    turn_all_lights_off()
                    set_traffic_light("NORTH")
                    time.sleep(25)
                    emergency_mode.value = False
                    pause_prediction.value = False
                    turn_all_lights_off()

# --- Traffic Light Thread ---
def normal_traffic_cycle():
    directions = ["NORTH", "EAST", "SOUTH","WEST"]
    current = 0
    while True:
        if not emergency_mode.value:
            set_traffic_light(directions[current])
            time.sleep(10)
            current = (current + 1) % 4

# --- Audio Detection (as a separate process) ---
def audio_process(siren_detected_time, yolo_detected_time, detection_trigger):
    import sounddevice as sd
    import tensorflow_hub as hub
    import tensorflow as tf
    import librosa

    yamnet_model = hub.load("yamnet_model")
    class_names = []
    with open("class.csv", "r") as f:
        next(f)
        for line in f:
            class_names.append(line.strip().split(',')[2])

    SAMPLE_RATE = 16000
    BLOCK_SECONDS = 3
    DURATION = SAMPLE_RATE * BLOCK_SECONDS
    audio_clip = []
    last_audio_detection_time = 0

    def callback(indata, frames, time_info, status):
        nonlocal audio_clip, last_audio_detection_time
        current_time = time.time()

        if current_time - last_audio_detection_time < 60:
            return

        audio_clip.extend(indata[:, 0])
        if len(audio_clip) >= DURATION:
            audio_np = np.array(audio_clip[:DURATION])
            audio_clip = audio_clip[DURATION:]
            audio_np = librosa.resample(audio_np, orig_sr=SAMPLE_RATE, target_sr=16000)
            scores, _, _ = yamnet_model(audio_np)
            predictions = tf.reduce_mean(scores, axis=0)
            top_class = tf.argmax(predictions).numpy()
            label = class_names[top_class].lower()
            if 'siren' in label or 'emergency' in label:
                print("Audio Detected: Siren")
                siren_detected_time.value = current_time
                last_audio_detection_time = current_time
                if dual_detection_triggered():
                    print("Dual detection confirmed (AUDIO then VIDEO)")
                    log_detection("DUAL")
                    detection_trigger.set()

    with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, callback=callback):
        while True:
            sd.sleep(1000)

# --- Start Threads & Processes ---
if __name__ == "__main__":

    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    model = YOLO("best.pt")
    cap = cv2.VideoCapture("http://192.168.161.120:4747/video")
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    threading.Thread(target=check_rfid, daemon=True).start()
    threading.Thread(target=normal_traffic_cycle, daemon=True).start()
    mp.Process(target=audio_process, args=(siren_detected_time, yolo_detected_time, detection_trigger), daemon=True).start()

    last_yolo_detection_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            current_time = time.time()

            if pause_prediction.value and current_time - yolo_detected_time.value > 60:
                pause_prediction.value = False

            if not pause_prediction.value and (current_time - last_yolo_detection_time > 60):
                results = model.predict(source=frame, imgsz=160, conf=0.6, stream=False, show=False)
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0].item())
                        label = model.names[cls_id].lower()
                        if label == "ambulance":
                            print("YOLO Detected: Ambulance")
                            yolo_detected_time.value = current_time
                            last_yolo_detection_time = current_time
                            pause_prediction.value = True

                            if dual_detection_triggered():
                                print("Dual detection confirmed (VIDEO then AUDIO)")
                                log_detection("DUAL")
                                detection_trigger.set()
                            else:
                                print("Waiting for siren...")

            if detection_trigger.is_set():
                if not emergency_mode.value:
                    emergency_mode.value = True
                    print("Emergency Mode Activated")
                    buzzer_alert()
                    turn_all_lights_off()
                    set_traffic_light("SOUTH")
                    time.sleep(25)
                    turn_all_lights_off()
                    emergency_mode.value = False
                    pause_prediction.value = False
                    detection_trigger.clear()

            cv2.imshow("Live YOLO", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('o'):
                print("Manual override activated - WEST GREEN")
                emergency_mode.value = True
                pause_prediction.value = True
                log_detection("MANUAL")
                buzzer_alert()
                turn_all_lights_off()
                set_traffic_light("WEST")
                time.sleep(25)
                turn_all_lights_off()
                emergency_mode.value = False
                pause_prediction.value = False

    except KeyboardInterrupt:
        cleanup()
    finally:
        cleanup()
