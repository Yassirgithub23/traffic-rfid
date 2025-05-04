import cv2
import time
import signal
import lgpio
import MFRC522
import queue
import numpy as np
import multiprocessing as mp
import threading
from ultralytics import YOLO

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
    "63C0AB1D": {"Name": "AMBULANCE 1", "Number": "TN 99 AB 0000"},
    "AA81E97A": {"Name": "AMBULANCE 2", "Number": "TN 99 AB 0001"},
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
            print(f"✅ {direction} GREEN")
            lgpio.gpio_write(chip, pins["GREEN"], 1)
            lgpio.gpio_write(chip, pins["RED"], 0)
        else:
            lgpio.gpio_write(chip, pins["GREEN"], 0)
            lgpio.gpio_write(chip, pins["RED"], 1)

def buzzer_alert(duration=2):
    print("🔔 Buzzer ON")
    lgpio.gpio_write(chip, BUZZER, 1)
    time.sleep(duration)
    lgpio.gpio_write(chip, BUZZER, 0)
    print("🔕 Buzzer OFF")

def cleanup():
    print("🧹 Cleaning up...")
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
                print(f"📡 RFID Detected: {uid_str}")
                if uid_str in card_names and not emergency_mode.value:
                    print(f"🚑 Authenticated: {card_names[uid_str]['Name']}")
                    emergency_mode.value = True
                    pause_prediction.value = True
                    buzzer_alert()
                    turn_all_lights_off()
                    set_traffic_light("NORTH")
                    time.sleep(15)
                    emergency_mode.value = False
                    pause_prediction.value = False
                    turn_all_lights_off()

# --- Traffic Light Thread ---
def normal_traffic_cycle():
    directions = ["NORTH", "EAST", "SOUTH"]
    current = 0
    while True:
        if not emergency_mode.value:
            set_traffic_light(directions[current])
            time.sleep(10)
            current = (current + 1) % 3

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

    def callback(indata, frames, time_info, status):
        nonlocal audio_clip
        audio_clip.extend(indata[:, 0])
        if len(audio_clip) >= DURATION:
            audio_np = np.array(audio_clip[:DURATION])
            audio_clip = audio_clip[DURATION:]
            audio_np = librosa.resample(audio_np, orig_sr=SAMPLE_RATE, target_sr=16000)
            scores, _, _ = yamnet_model(audio_np)
            predictions = tf.reduce_mean(scores, axis=0)
            top_class = tf.argmax(predictions).numpy()
            label = class_names[top_class].lower()
            if 'siren' in label:
                print("🚨 Audio Detected: Siren")
                siren_detected_time.value = time.time()
                if dual_detection_triggered():
                    print("✅ Dual detection confirmed (AUDIO then VIDEO)")
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

    # Camera and YOLO
    model = YOLO("best.pt")
    cap = cv2.VideoCapture("http://192.168.174.135:4747/video")
    if not cap.isOpened():
        print("❌ Cannot open camera")
        exit()

    threading.Thread(target=check_rfid, daemon=True).start()
    threading.Thread(target=normal_traffic_cycle, daemon=True).start()
    mp.Process(target=audio_process, args=(siren_detected_time, yolo_detected_time, detection_trigger), daemon=True).start()

    last_yolo_detect_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # Check pause state
            if pause_prediction.value and time.time() - yolo_detected_time.value > 45:
                pause_prediction.value = False

            if not pause_prediction.value:
                results = model.predict(source=frame, imgsz=160, conf=0.6, stream=False, show=False)
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0].item())
                        label = model.names[cls_id].lower()
                        if label == "ambulance":
                            yolo_detected_time.value = time.time()
                            print("🚑 YOLO Detected: Ambulance")
                            pause_prediction.value = True
                            model.predict(source=frame, conf=1.0)  # Pause further detection

                            if dual_detection_triggered():
                                print("✅ Dual detection confirmed (VIDEO then AUDIO)")
                                detection_trigger.set()
                            else:
                                print("⚠️ Waiting for siren...")

            if detection_trigger.is_set():
                if not emergency_mode.value:
                    emergency_mode.value = True
                    print("🚨 Emergency Mode Activated")
                    buzzer_alert()
                    turn_all_lights_off()
                    set_traffic_light("NORTH")
                    time.sleep(15)
                    turn_all_lights_off()
                    emergency_mode.value = False
                    pause_prediction.value = False
                    detection_trigger.clear()

            cv2.imshow("Live YOLO", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        cleanup()
    finally:
        cleanup()
