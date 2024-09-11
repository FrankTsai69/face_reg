import cv2 as cv
import time

#test project
import yunet
backend_target_pairs=[cv.dnn.DNN_BACKEND_OPENCV,cv.dnn.DNN_TARGET_CPU]
model_d=yunet.YuNet(modelPath='./model/face_detection/face_detection_yunet_2023mar.onnx',
			inputSize=[320,320],
			confThreshold=0.9,#信心閾值
			nmsThreshold=0.3,#bbox閾值
			topK=5000,#top_k(NMS的輸出分數須高於該值才會被判斷為物體)
			backendId=backend_target_pairs[0],
			targetId=backend_target_pairs[1])
tt=1#open test-time mode

#test project
cap=cv.VideoCapture(0)
opened=cap.isOpened()
tm=cv.TickMeter()

if opened:
	print("detected camera")
	
	w=int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
	h=int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
	
	#test project
	
	model=cv.FaceDetectorYN.create(
            model='./model/face_detection/face_detection_yunet_2023mar.onnx',
            config="",
            input_size=[w,h],
            score_threshold=0.9,
            nms_threshold=0.3,
            top_k=5000,
            backend_id=backend_target_pairs[0],
			target_id=backend_target_pairs[1])
	model_d.setInputSize([w,h])
	cd=0
	#test project
	
	while 1:
		opened,frame=cap.read()
		frame=cv.flip(frame,1)
		#test time
		if tt==1:
			faces =model.detect(frame)
			#sult=model_d.infer(frame)
		#test time
		
		cv.imshow('test',frame)
		if cv.waitKey(1)&0xff==ord('q'):
			print("")
			cv.destroyAllWindows()
			cap.release()
			break
		
