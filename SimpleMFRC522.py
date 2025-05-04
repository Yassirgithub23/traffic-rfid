#!/usr/bin/env python
# -*- coding: utf8 -*-
#
#    Modified by Mohammed Yassir for Raspberry Pi 5 with lgpio
#    Originally written by Simon Monk
#

import MFRC522
import lgpio as GPIO
import time

class SimpleMFRC522:

    READER = None
    KEY = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]
    BLOCK_ADDRS = [8, 9, 10]
    PIN_RST = 29

    def __init__(self):
        self.READER = MFRC522()
        self.chip = GPIO.gpiochip_open(0)
        GPIO.gpio_claim_output(self.chip, self.PIN_RST)
        GPIO.gpio_write(self.chip, self.PIN_RST, 1)
        time.sleep(0.1)
        GPIO.gpio_write(self.chip, self.PIN_RST, 0)
        time.sleep(0.1)
        GPIO.gpio_write(self.chip, self.PIN_RST, 1)

    def read(self):
        id, text = self.read_no_block()
        while not id:
            id, text = self.read_no_block()
        return id, text

    def read_id(self):
        id = self.read_id_no_block()
        while not id:
            id = self.read_id_no_block()
        return id

    def read_id_no_block(self):
        # Request tag
        (status, TagType) = self.READER.MFRC522_Request(self.READER.PICC_REQIDL)
        if status != self.READER.MI_OK:
            return None
        
        # Anti-collision
        (status, uid) = self.READER.MFRC522_Anticoll()
        if status != self.READER.MI_OK:
            return None
        
        return self.uid_to_num(uid)

    def read_no_block(self):
        # Request card presence
        (status, TagType) = self.READER.MFRC522_Request(self.READER.PICC_REQIDL)
        if status != self.READER.MI_OK:
            return None, None
        
        # Prevent card collision
        (status, uid) = self.READER.MFRC522_Anticoll()
        if status != self.READER.MI_OK:
            return None, None
        
        # Convert UID to number
        id = self.uid_to_num(uid)
        
        # Select the card
        self.READER.MFRC522_SelectTag(uid)
        
        # Authenticate the card with key
        status = self.READER.MFRC522_Auth(self.READER.PICC_AUTHENT1A, 11, self.KEY, uid)
        if status != self.READER.MI_OK:
            self.READER.MFRC522_StopCrypto1()
            return id, None

        # Read the block data
        data = []
        text_read = ''
        for block_num in self.BLOCK_ADDRS:
            block = self.READER.MFRC522_Read(block_num)
            if block:
                data += block

        # Convert the data to text
        if data:
            text_read = ''.join(chr(i) for i in data if i != 0)
        
        # Stop encryption
        self.READER.MFRC522_StopCrypto1()
        return id, text_read
    
    def write(self, text):
        id, text_in = self.write_no_block(text)
        while not id:
            id, text_in = self.write_no_block(text)
        return id, text_in

    def write_no_block(self, text):
        # Request tag
        (status, TagType) = self.READER.MFRC522_Request(self.READER.PICC_REQIDL)
        if status != self.READER.MI_OK:
            return None, None
        
        # Anti-collision
        (status, uid) = self.READER.MFRC522_Anticoll()
        if status != self.READER.MI_OK:
            return None, None
        
        # Convert UID to number
        id = self.uid_to_num(uid)
        
        # Select the card
        self.READER.MFRC522_SelectTag(uid)
        
        # Authenticate
        status = self.READER.MFRC522_Auth(self.READER.PICC_AUTHENT1A, 11, self.KEY, uid)
        if status != self.READER.MI_OK:
            self.READER.MFRC522_StopCrypto1()
            return None, None
        
        # Convert text to bytes
        data = bytearray()
        data.extend(bytearray(text.ljust(len(self.BLOCK_ADDRS) * 16).encode('ascii')))
        
        # Write to block
        i = 0
        for block_num in self.BLOCK_ADDRS:
            self.READER.MFRC522_Write(block_num, data[i*16:(i+1)*16])
            i += 1
        
        # Stop communication
        self.READER.MFRC522_StopCrypto1()
        return id, text[0:(len(self.BLOCK_ADDRS) * 16)]

    def uid_to_num(self, uid):
        n = 0
        for i in range(0, 5):
            n = n * 256 + uid[i]
        return n
