import lgpio
from mfrc522 import SimpleMFRC522
import time

# Initialize the GPIO chip and RFID reader
chip = lgpio.gpiochip_open(0)
reader = SimpleMFRC522()

try:
    text = input("Enter new data: ")
    print("Place your tag to write...")
    reader.write(text)
    print("Data written successfully!")
    time.sleep(2)

except KeyboardInterrupt:
    print("Process stopped by user.")

finally:
    lgpio.gpiochip_close(chip)
    GPIO.gpiochip_close(0)
