

def c():
	start=time.time()
	end=time.time()
	#mo=end-start
	thread3=threading.Thread(target=get_time,args=('',q_time))
	thread3.daemon=True
	thread4=threading.Thread(target=get_time,args=('',q_time))
	thread4.daemon=True
	thread3.start()
def a():
	for i in range(5):
		a=10
	#q=queue.Queue()
	#q.put('a')
	#b=q.get()
	tm=cv.TickMeter()
	tm.start()
	start=time.time()
	end=time.time()
	c=start-end
	tm.stop()
	fps=tm.getFPS()
	tm.reset()
def b():
	for i in range(5):
		a=10
	tm=cv.TickMeter()
	#tm=cv.TickMeter()
	#tm.start()
	#start=time.time()
	#end=time.time()
	#tm.stop()
	#fps=tm.getFPS()
	#tm.reset()
	
if __name__=='__main__':
	import timeit
	import time
	import cv2 as cv
	import queue
	#c=timeit.timeit('c()',setup='from __main__ import c',number=1)
	#print(c)
	a=timeit.timeit('a()',setup="from __main__ import a",number=1000)
	print("a:",a)
	b=timeit.timeit('b()',setup="from __main__ import b",number=1000)
	print("b:",b)
	#print(b-a-c)
	
	
