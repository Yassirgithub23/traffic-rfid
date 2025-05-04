import time
import lgpio
import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("best.pt")

# Define GPIO pins
RED_LED = 27
GREEN_LED = 17
BUZZER = 18

# Open GPIO chip
chip = lgpio.gpiochip_open(0)

# Setup GPIOs as output
lgpio.gpio_claim_output(chip, RED_LED)
lgpio.gpio_claim_output(chip, GREEN_LED)
lgpio.gpio_claim_output(chip, BUZZER)

ambulance_detected_once = False

# Open the camera (use 0 for USB cam, or IP webcam stream URL)
cap = cv2.VideoCapture("http://192.168.115.135:4747/video")  # Replace <your-ip>

def detect_ambulance(frame):
    results = model.predict(source=frame, show=False, verbose=False)

    for result in results:
        for box in result.boxes:
            label = model.names[int(box.cls[0].item())]
            conf = box.conf[0].item()
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if label.lower() == "ambulance":
                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                print("?? Ambulance Detected!")
                return True, frame

    return False, frame

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("? Failed to get frame")
            break

        # Process only one frame every second
        detected, frame = detect_ambulance(frame)

        if detected and not ambulance_detected_once:
            ambulance_detected_once = True

            # Activate buzzer
            lgpio.gpio_write(chip, BUZZER, 1)
            time.sleep(3)
            lgpio.gpio_write(chip, BUZZER, 0)

            # Set green light for ambulance
            lgpio.gpio_write(chip, RED_LED, 0)
            lgpio.gpio_write(chip, GREEN_LED, 1)
            time.sleep(45)

            ambulance_detected_once = False
        else:
            # Normal cycle
            lgpio.gpio_write(chip, RED_LED, 1)
            lgpio.gpio_write(chip, GREEN_LED, 0)
            time.sleep(5)
            lgpio.gpio_write(chip, RED_LED, 0)
            lgpio.gpio_write(chip, GREEN_LED, 1)
            time.sleep(5)

        # Show the detection window
        cv2.imshow("Ambulance Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(1)  # wait 1 second before capturing next frame

except KeyboardInterrupt:
    print("?? Program interrupted")

finally:
    lgpio.gpio_write(chip, RED_LED, 0)
    lgpio.gpio_write(chip, GREEN_LED, 0)
    lgpio.gpio_write(chip, BUZZER, 0)
    lgpio.gpiochip_close(chip)
    cap.release()
    cv2.destroyAllWindows()
    print("? Cleaned up resources.")
