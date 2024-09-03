import RPi.GPIO as GPIO
import time
#define tri's and echo's pin
trig=23
echo=24
#init
GPIO.setmode(GPIO.BCM)
GPIO.setup(trig,GPIO.OUT)#trig
GPIO.setup(echo,GPIO.IN)#echo



def sonic():
	count=[]
	for i in range(3):
		#send tri pin 10us
		GPIO.output(trig,False)
		time.sleep(0.1)
		GPIO.output(trig,True)
		time.sleep(0.00001)
		GPIO.output(trig,False)
		while GPIO.input(echo)==0:
			pulse_start=time.time()
		while GPIO.input(echo)==1:
			pulse_end=time.time()
		pulse_duration=pulse_end-pulse_start
		distance=pulse_duration*17000
		if pulse_duration >=0.01746:
			print("time out")
			return False,-1
		elif distance>300 or distance==0:
			print('out of range')
			return False,-2
		distance=round(distance,3)
		print('Distnace: %f cm'%distance)
		count.append(distance)
	return True,sum(count)/3
	

