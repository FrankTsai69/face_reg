import cv2 as cv
cap=cv.VideoCapture(0)
opened=cap.isOpened()
if opened:
	print("detected camera")
	
	w=int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
	h=int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
	cap.set(3,1100)
	cap.set(4,100)
	cap.set(5,60)
	while 1:
		opened,frame=cap.read()
		
		cv.imshow('test',frame)
		if cv.waitKey(1)&0xff==ord('q'):
			cv.destroyAllWindows()
			cap.release()
			break
		
