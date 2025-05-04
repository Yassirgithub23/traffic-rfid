import RPi.GPIO as GPIO
from mfrc522 import SimpleMFRC522

reader = SimpleMFRC522()

try:
    print("Please your card near the reader")
    id, text = reader.read()
    print("Id : %s\n Text: %s" %(id,text))
finally:
    GPIO.cleanup()
