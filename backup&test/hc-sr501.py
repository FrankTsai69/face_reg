import RPi.GPIO as GPIO
import time
from datetime import datetime


pir=18# hc-sr501
GPIO.setmode(GPIO.BCM)
GPIO.setup(pir,GPIO.IN)
while True:
	input_state =GPIO.input(pir)
	if input_state==True:
		print("detected...")
		time.sleep(0.2)
	
