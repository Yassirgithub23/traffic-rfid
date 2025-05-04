#!/usr/bin/env python
# -*- coding: utf8 -*-
#
#    Corrected by Mohammed Yassir for Raspberry Pi 5 with lgpio
#    Originally developed by Mario Gomez <mario.gomez@teubi.co>
#

import lgpio as GPIO
import spidev
import time
import logging

class MFRC522:
    MAX_LEN = 16

    PCD_IDLE = 0x00
    PCD_AUTHENT = 0x0E
    PCD_RECEIVE = 0x08
    PCD_TRANSMIT = 0x04
    PCD_TRANSCEIVE = 0x0C
    PCD_RESETPHASE = 0x0F
    PCD_CALCCRC = 0x03

    PICC_REQIDL = 0x26
    PICC_REQALL = 0x52
    PICC_ANTICOLL = 0x93
    PICC_SELECTTAG = 0x93
    PICC_AUTHENT1A = 0x60
    PICC_AUTHENT1B = 0x61
    PICC_READ = 0x30
    PICC_WRITE = 0xA0
    PICC_HALT = 0x50

    MI_OK = 0
    MI_NOTAGERR = 1
    MI_ERR = 2

    def __init__(self, bus=0, device=0, spd=1000000, pin_rst=22, debugLevel='WARNING'):
        self.spi = spidev.SpiDev()
        self.spi.open(bus, device)
        self.spi.max_speed_hz = spd

        self.logger = logging.getLogger('mfrc522Logger')
        self.logger.addHandler(logging.StreamHandler())
        level = logging.getLevelName(debugLevel)
        self.logger.setLevel(level)

        # Setup GPIO using lgpio
        self.chip = GPIO.gpiochip_open(0)
        GPIO.gpio_claim_output(self.chip, pin_rst)

        # Reset the RFID module
        GPIO.gpio_write(self.chip, pin_rst, 1)
        time.sleep(0.1)
        GPIO.gpio_write(self.chip, pin_rst, 0)
        time.sleep(0.1)
        GPIO.gpio_write(self.chip, pin_rst, 1)

        self.MFRC522_Init()

    def MFRC522_Reset(self):
        self.Write_MFRC522(0x01, self.PCD_RESETPHASE)

    def Write_MFRC522(self, addr, val):
        self.spi.xfer2([(addr << 1) & 0x7E, val])

    def Read_MFRC522(self, addr):
        val = self.spi.xfer2([((addr << 1) & 0x7E) | 0x80, 0])
        return val[1]

    def Close_MFRC522(self):
        self.spi.close()
        GPIO.gpiochip_close(self.chip)

    def SetBitMask(self, reg, mask):
        tmp = self.Read_MFRC522(reg)
        self.Write_MFRC522(reg, tmp | mask)

    def ClearBitMask(self, reg, mask):
        tmp = self.Read_MFRC522(reg)
        self.Write_MFRC522(reg, tmp & (~mask))

    def AntennaOn(self):
        self.SetBitMask(0x14, 0x03)

    def AntennaOff(self):
        self.ClearBitMask(0x14, 0x03)

    def MFRC522_Request(self, reqMode):
        self.Write_MFRC522(0x0D, 0x07)
        (status, backData, backBits) = self.MFRC522_ToCard(self.PCD_TRANSCEIVE, [reqMode])
        if ((status != self.MI_OK) | (backBits != 0x10)):
            status = self.MI_ERR
        return status, backData

    def MFRC522_Anticoll(self):
        serNum = [self.PICC_ANTICOLL, 0x20]
        (status, backData, backBits) = self.MFRC522_ToCard(self.PCD_TRANSCEIVE, serNum)
        if status == self.MI_OK:
            return backData
        else:
            return None

    def MFRC522_Auth(self, authMode, BlockAddr, Sectorkey, serNum):
        buff = [authMode, BlockAddr] + Sectorkey + serNum[:4]
        (status, backData, backLen) = self.MFRC522_ToCard(self.PCD_AUTHENT, buff)
        return status

    def MFRC522_Read(self, blockAddr):
        recvData = [self.PICC_READ, blockAddr]
        (status, backData, backLen) = self.MFRC522_ToCard(self.PCD_TRANSCEIVE, recvData)
        return backData

    def MFRC522_Write(self, blockAddr, writeData):
        buff = [self.PICC_WRITE, blockAddr] + writeData
        (status, backData, backLen) = self.MFRC522_ToCard(self.PCD_TRANSCEIVE, buff)
        return status

    def MFRC522_StopCrypto1(self):
        self.ClearBitMask(0x08, 0x08)

    def MFRC522_ToCard(self, command, sendData):
        backData = []
        self.Write_MFRC522(0x01, self.PCD_IDLE)
        for data in sendData:
            self.Write_MFRC522(0x09, data)
        self.Write_MFRC522(0x01, command)

        # Wait for the response
        i = 1000
        while True:
            n = self.Read_MFRC522(0x04)
            i -= 1
            if n & 0x30:
                break
            if i == 0:
                return self.MI_ERR, None, 0

        if n & 0x01:
            return self.MI_NOTAGERR, None, 0

        backLen = self.Read_MFRC522(0x0A)
        backData = [self.Read_MFRC522(0x09) for _ in range(backLen)]
        return self.MI_OK, backData, backLen

    def MFRC522_Init(self):
        self.MFRC522_Reset()
        self.Write_MFRC522(0x11, 0x3D)
        self.Write_MFRC522(0x14, 0x40)
        self.AntennaOn()
