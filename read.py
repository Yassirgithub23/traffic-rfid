import SimpleMFRC522

reader = SimpleMFRC522.SimpleMFRC522()

try:
    print("Place your RFID card near the reader...")
    id, text = reader.read()
    print(f"Card ID: {id}")
    print(f"Text: {text}")
except KeyboardInterrupt:
    print("Process interrupted.")
finally:
    print("Cleaning up GPIO.")
    reader.READER.Close_MFRC522()
