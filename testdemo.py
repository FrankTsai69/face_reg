import numpy as np
import cv2 as cv 
import pandas as pd
import os
from function import yunet
from function import face_visualize as fv
from function import face_feature as ff
import threading
from time import sleep
from queue import Queue
def print_check(name=None):
    if name != None:
        print(f"name:{name[1]['name']} ,norm:{name[1]['score']:.2f}")
    elif name == None:
        print("")
def c(frame):
    cv.imshow("a",frame)
def f(model_d,model_r,frame,q,match_feature):
        match_meth=1
        name={}
        #人臉偵測
        results=model_d.infer(frame)
        if results.shape[0] == 1:
            if results.shape[1] ==15: 
                name[1]={"results":results[0],"feature":[],"name":"none","score":0}
                #特徵值擷取
                name=ff.feature(model_r,frame,name)
                #比較
                name=ff.match(model_r,name,match_feature,match_meth,0.9)  
                print(name[1]['name'])   
                if name[1]['name'] != 'unknown':
                    output = frame.copy()
                    box_color=(0, 255, 0)
                    text_color=(0, 0, 255)
                    det=name[1]["results"]
                    #bbox框
                    bbox = det[0:4].astype(np.int32)
                    cv.rectangle(output, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), box_color, 2)
                    #名稱
                    cv.putText(output, '{}'.format(name[1]["name"]), (bbox[0], bbox[1]-10), cv.FONT_HERSHEY_DUPLEX, 0.5, text_color)
                    print_check(name)

                    q.put([1,output])
                else:
                    print_check(name)
        else:
            print("")
        q.put([0,None])

def main():
    print("init...")
    data_path='./data/test.pkl'
    print("loading MatchDatabase...")    
    match_feature=pd.read_pickle(data_path)
    print("---DataBase loading complete---")
    print("DataBase count:",len(match_feature))
    
    print("loading Model...")
    #運行位置
    backend_target_pairs=[cv.dnn.DNN_BACKEND_OPENCV,cv.dnn.DNN_TARGET_CPU]
    print("loading Face Detection Model...")
    print("---loading Model 1/2---")
    #人臉偵測模組
    model_d=yunet.YuNet(modelPath='./model/face_detection/face_detection_yunet_2023mar.onnx',
                inputSize=[320,320],
                confThreshold=0.9,#信心閾值
                nmsThreshold=0.3,#bbox閾值
                topK=5000,#top_k(NMS的輸出分數須高於該值才會被判斷為物體)
                backendId=backend_target_pairs[0],
                targetId=backend_target_pairs[1])
    print("loading Face Recognition Model...")
    #人臉辨識模組
    model_r=cv.FaceRecognizerSF.create(model='./model/face_recognition/face_recognition_sface_2021dec.onnx',
                                      config="",
                                      backend_id=backend_target_pairs[0],
                                      target_id=backend_target_pairs[1])
    print("---loading Model 2/2---")
    print("---Model loading complete---")
    
    #輸入串流並確認是否接收到設備
    if cv.VideoCapture(0).isOpened():
        
        time=0
        # time=0
        print("已偵測到攝影機...")
        #建立設備實體
        cap=cv.VideoCapture(0)
        tm=cv.TickMeter()
        #讀取當前frame尺寸640*480
        w=int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        h=int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        print("螢幕尺寸:",w,"x",h,"pi")
        model_d.setInputSize([w,h])
        count=0
        q=Queue()  
        while 1:
            #hasFrame:讀取是否成功,frame:讀取影像
            hasFrame,frame=cap.read()
            if not hasFrame:
                print("未偵測到影像")
                cv.destroyAllWindows()
                break
            frame=cv.flip(frame, 1)
            thread=threading.Thread(target=f,args=(model_d,model_r,frame,q,match_feature))
            thread2=threading.Thread(target=c,args=[frame])
            if q.empty and count > 200:        
                thread.start()
                ls=[]
                ls=q.get()
                print(ls[0])
                if ls[0] == 1:
                    cv.imshow("face detection",ls[1])
                    cv.waitKey(1)
                    sleep(1)
                    
                q.queue.clear()
                count=0
            cv.imshow('face detection',frame)
            if cv.waitKey(1)&0xff == ord("q"):
                cv.destroyAllWindows()
                print("---end---")
                break
            count+=1

if __name__=='__main__':
    main()
