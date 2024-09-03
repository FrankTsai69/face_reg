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
import RPi.GPIO as GPIO

def print_check(name=None,score=None,state=None):
    print(f"""\rstate: name: norm:                           """,end="")
    if name != None and state=="found":
        print(f"""\rstate:{state} name:{name} norm:{score:.2f}""",end="")
    elif name == None and state !="found":
        print(f"""\rstate:not found name: norm:""",end="")
    elif name == None and state=="found":
        print(f"""\rstate:{state} name:unknow norm:-1 """,end="")
def c(frame):
    cv.imshow("a",frame)
def f(model_d,model_r,frame,q,match_feature):
        match_meth=1
        name={}
        #人臉偵測
        results=model_d.infer(frame)
        if results.shape[0] == 1 :
            x=results[0][0]#0~640
            y=results[0][1]#0~480
            if 15<x<625 and 15<y<465:
                if results.shape[1] ==15: 
                    name[1]={"results":results[0],"feature":[],"name":"none","score":0}
                    #特徵值擷取
                    name=ff.feature(model_r,frame,name)
                    #比較
                    name=ff.match(model_r,name,match_feature,match_meth,0.9)  
                     
                    if name[1]['name'] != 'unknown':
                        output=fv.visualize(frame,name,mode=0)
                        print_check(name=name[1]['name'],score=name[1]['score'],state='found')

                        q.put([1,output])
                    else:
                        print_check()                    
        q.put([0,None])

def main():
    print("init...")
    print("---loading variable---",end="")
    time=0
    delaytime=50
    count=delaytime/2
    q=Queue()  
    print("\t done.")
    print("---loading HC-SR501---",end="")
    
    pir=18# hc-sr501
    led=23#
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(pir,GPIO.IN)
    GPIO.setup(led,GPIO.OUT)
    print("\t done.")
    
    print("---loading MatchDatabase---",end="")  
    data_path='./data/test.pkl'  
    match_feature=pd.read_pickle(data_path)
    print("\t done.")
    print("DataBase count:",len(match_feature))
    
    print("loading Model...")
    #運行位置
    backend_target_pairs=[cv.dnn.DNN_BACKEND_OPENCV,cv.dnn.DNN_TARGET_CPU]
    print("---loading Model 1/2---")
    print("loading Face Detection Model",end="")
    #人臉偵測模組
    model_d=yunet.YuNet(modelPath='./model/face_detection/face_detection_yunet_2023mar.onnx',
                inputSize=[320,320],
                confThreshold=0.9,#信心閾值
                nmsThreshold=0.3,#bbox閾值
                topK=5000,#top_k(NMS的輸出分數須高於該值才會被判斷為物體)
                backendId=backend_target_pairs[0],
                targetId=backend_target_pairs[1])
    print('\t done.')
    print("confTgreshold:",0.9)
    print("nmsTgreshold:",0.3)
    print("topK:",5000)
    
    print("---loading Model 2/2---")
    print("loading Face Recognition Model",end="")
    #人臉辨識模組
    model_r=cv.FaceRecognizerSF.create(model='./model/face_recognition/face_recognition_sface_2021dec.onnx',
                                      config="",
                                      backend_id=backend_target_pairs[0],
                                      target_id=backend_target_pairs[1])
    print('\t done.')
    print("---Model loading complete---")
    
    #輸入串流並確認是否接收到設備
    if cv.VideoCapture(0).isOpened():
        
        print("已偵測到攝影機...")
        print("---setting carman---",end="")
        #建立設備實體
        cap=cv.VideoCapture(0)
        tm=cv.TickMeter()
        cap.set(5,60)
        #讀取當前frame尺寸640*480
        cap.set(3,640)
        cap.set(4,480)
        w=int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        h=int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        
        model_d.setInputSize([w,h])
        print('\t done.')
        print("螢幕尺寸:",w,"x",h,"pi")
        print('init done...')
        
        while 1:
            
            input_state =GPIO.input(pir)
            #hasFrame:讀取是否成功,frame:讀取影像
            hasFrame,frame=cap.read()
            if not hasFrame:
                print("")
                print("未偵測到影像")
                print("closing window",end="")
                cv.destroyAllWindows()
                print("---done.")
                print("closing cam",end="")
                cap.release()
                print("---done.")
                GPIO.output(led,False)
                
                print("---end---")
                break

            frame=cv.flip(frame, 1)
            
            if input_state==True :
                GPIO.output(led,True)
                frame=fv.visualize(frame,mode=2,size=(w,h))
                thread=threading.Thread(target=f,args=(model_d,model_r,frame,q,match_feature))
                print_check(state="found")
                if q.empty and count > delaytime:   
                     
                    thread.start()
                    ls=[]
                    ls=q.get()
                    if ls[0] == 1:
                        cv.imshow("face detection",ls[1])
                        cv.waitKey(1)
                        sleep(1)
                    q.queue.clear()
                    count=0
            else:
                print_check(state='')
                GPIO.output(led,False)
            cv.imshow('face detection',frame)
            if cv.waitKey(1)&0xff == ord("q"):
                print("")
                print("closing window",end="")
                cv.destroyAllWindows()
                print("---done.")
                print("closing cam",end="")
                cap.release()
                print("---done.")
                GPIO.output(led,False)
                
                print("---end---")
                break
            count+=1


if __name__=='__main__':
    main()
