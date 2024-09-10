import numpy as np
import os
import cv2 as cv
import pandas as pd
import pickle
import function.face_visualize as fv
import RPi.GPIO as GPIO
import threading
import time
from queue import Queue
matchdata={}
save_path='./data/feature.pkl'
def btn_event(btn,q2):
    while True:
        if GPIO.input(btn) and q2.empty():
            start=time.time()
            while GPIO.input(btn):
                pass
            end=time.time()
            times=end-start
            if times>2:
                q2.put([2])
                
            elif 0<times<2:
                q2.put([1])
            time.sleep(0.5)
def f(model_d,model_r,frame,q):
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
                     
                    if name[1]['name'] != 'unknown':
                        output=fv.visualize(frame,name,mode=0)
                        print_check(name=name[1]['name'],score=name[1]['score'],state='found')
                        now=datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
                        sa.main(now,name[1]['name'])
                        q.put([1,output])
                    else:
                        print_check()                    
        q.put([0,None])
def set_data(model_d,model_r):
    if "feature.pkl" in os.listdir('./data'):
        matchdata=pd.read_pickle(save_path)
    q=Queue()
    q2=Queue()
    btn=23
    thread=threading.Thread(target=btn_event,args=(btn,q2))
    name={}
    check=""
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(btn,GPIO.IN,pull_up_down=GPIO.PUD_DOWN)
    thread.daemon=True
    thread.start()
    cap=cv.VideoCapture(0)
    if cap.isOpened():
        print("已偵測到攝影機...")
        print("---setting carman---",end="")
        #建立設備實體
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
        while 1:
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
                flag=False
                print("---end---")
                break

            frame=cv.flip(frame, 1)
            cv.imshow('face detection',frame)
            if cv.waitKey(1)&0xff == ord("q"):
                print("")
                print("closing window",end="")
                cv.destroyAllWindows()
                print("---done.")
                print("closing cam",end="")
                cap.release()
                print("---done.")
                
                print("---end---")
                exit()
            if cv.waitKey(1)&0xff == ord(" "):
                print("")
                print("closing window",end="")
                cv.destroyAllWindows()
                print("---done.")
                print("closing cam",end="")
                cap.release()
                print("---done.")
                
                print("---end---")
                break
            if not q2.empty():
                event=q2.get()[0]
                print("event:",event)
                if event==1:
                    results=model_d.infer(frame)
                    if results.shape[1]==15:
                        name[1]={"results":results[0],"feature":[]}
                        output=fv.visualize(frame, name,mode=0)
                    else:
                        print("do not detection face")
                        q2.queue.clear()
                        continue
                    
                    cv.imshow('face detection',output)
                    cv.waitKey(1)
                    
                    target=input("請輸入名稱:")
                    if target in matchdata.keys():
                        check=input("name already exist,are you sure to save?(Y/N): ").lower()
                        while not check in ["yes","no","n","y"]:
                            
                            check=input("please enter Yes or No: ").lower()
                        if check in ["no","n"]:
                            continue
                    input_r=model_r.alignCrop(frame, results)
                    feature_r=model_r.feature(input_r)
                    matchdata[target]=feature_r
                    pd.to_pickle(matchdata,save_path)

                    print("檔案建立完成")
                elif event==2:
                    print("leave set_data mode")
                    print("closing window",end="")
                    cv.destroyAllWindows()
                    print("---done.")
                    print("closing cam",end="")
                    cap.release()
                    print("---done.")
                    
                    print("---end---")
                    break
            

