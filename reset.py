from mfrc522 import SimpleMFRC522
import lgpio as GPIO
import time

# Initialize the RFID Reader
reader = SimpleMFRC522()

print("?? Place your RFID card near the reader to format it...")
time.sleep(2)

try:
    # Read the card UID first
    id, text = reader.read()
    print(f"? Card Detected: ID - {id}")
    print(f"?? Current Data: {text}")

    # Completely erase data (write blank spaces)
    print("?? Formatting the card...")
    blank_data = " " * 48  # 3 blocks of 16 bytes each

    # Write blank data to all 3 blocks
    reader.write(blank_data)
    print("? Card formatted successfully!")
    print("?? All previous data has been erased.")

except Exception as e:
    print(f"? Error: {e}")

finally:
    # Clean up GPIO
    GPIO.cleanup()
    print("?? You can now write new data to this card.")
