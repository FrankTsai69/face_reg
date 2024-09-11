#! /home/12/py_envs/bin/python3
import numpy as np
import cv2 as cv 
import pandas as pd
import os
from function import yunet
from function import face_visualize as fv
from function import face_feature as ff
from function import SaveAbsent as sa
import threading
import time
import datetime
from queue import Queue
import RPi.GPIO as GPIO

class color():
    #print(color.red+message+color.reset)
    red='\033[31m'
    green='\033[32m'
    reset='\033[0m'
def InitDivider(s,num=53,indent=0):
    c=len(s)
    if (num-c)%2==0:
        print(" "*indent+'-'*int((num-c)/2)+s+" "+'-'*int((num-c)/2))
    elif (num-c)%2==1:
        print(" "*indent+'-'*int((num-c+1)/2)+s+""+'-'*int((num-c-1)/2))
def InitResults(results):
    if results:
        print(color.green+'\t done'+color.reset)
    else:
        print(color.red+'\t false'+color.reset)
def print_check(name=None,score=None,state=None,r_cd=None,i_cd=None,HCstate=None,s_cd=None):
    dict1={'state':str(state),'name':str(name),'score':str(score),'r_cd':str(r_cd),'i_cd':str(i_cd),'s_cd':str(s_cd),'HCstate':str(HCstate)}    
    print("\033[A\x1b[2K"*(len(dict1.keys())+1))
    
    for i in range(len(dict1.keys())):
        k=list(dict1.keys())[i]
        print(k+": "+dict1[k])
def buttcheck(btn,q_btn):
    while 1:
        snd=False
        fir=False
        if GPIO.input(btn) and q_btn.empty():
            start=time.time()
            while GPIO.input(btn):
                fir=True
            end=time.time()
            time.sleep(0.2)
            while GPIO.input(btn):
                snd=True
            end2=time.time()
            if end-start>2 or end2-end>2:
                q_btn.put([2])
            elif fir==True and snd == True:
                q_btn.put([3])
                time.sleep(0.3)

def f(model_d,model_r,frame,q_r,match_feature):
        match_meth=1
        name={}
        #人臉偵測
        results=model_d.infer(frame)
        if results.shape[0] == 1 :
            x=results[0][0]#0~640
            y=results[0][1]#0~480
            if 60<x<580 and 60<y<420:
                if results.shape[1] ==15: 
                    name[1]={"results":results[0],"feature":[],"name":"none","score":0}
                    #特徵值擷取
                    name=ff.feature(model_r,frame,name)
                    #比較
                    name=ff.match(model_r,name,match_feature,match_meth,0.9)  
                     
                    if name[1]['name'] != 'unknown':
                        #output=fv.visualize(frame,name,mode=0)
                        
                        now=datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
                        sa.main(now,name[1]['name'])
                        q_r.put([1,frame,name[1]['name'],name[1]['score']])
                
            else: q_r.put([-1,frame,"Please stand in front of the camera!"])
        elif results.shape[0]>1: q_r.put([-1,frame,"Too many people,unable to identify"])
        q_r.put([0,None])

def main():
    InitDivider("start init",60)
    init_list=[False,False,False,False,False,False,False,False,False]

    InitDivider('loading variable')
    print(f"{'time':40s}",end="")
    #time :init_list 0
    try:
        cd_set=[[2,3],[15,10,5],3]#second;index:0 for recognizer,1 for infrared(15:duration time,10:cooldown time)
        ct=-1 #cycle time
        r_cd=3 #recognizer cooldown
        i_cd=3 #infrared cooldown
        s_cd=cd_set[2] #show cooldown
    except:
        InitResults(False)
    else:
        InitResults(True)
        init_list[0]=True

    InitDivider('loading object')
    print(f"{'queue':40s}",end='')
    #queue :init_list 1
    try:
        ls=[0]
        q_r=Queue()
        q_btn=Queue()
    except:
        InitResults(False)
    else:
        InitResults(True)
        init_list[1]=True

    print(f"{'GPIO':40s}",end="")
    #GPIO :init_list 2
    try:
        #hc-sr501
        HCstate=0# hc-sr501
        HC=18# hc-sr501
        #btn
        btn=23# close button
        #GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(HC,GPIO.IN)
        GPIO.setup(btn,GPIO.IN,pull_up_down=GPIO.PUD_DOWN)
    except:
        InitResults(False)
    else:
        InitResults(True)
        init_list[2]=True
    
    print(f'{"threading":40s}',end='')    
    #threading :init_list 3
    try:
        thread2=threading.Thread(target=buttcheck,args=(btn,q_btn))
        thread2.daemon=True
    except:
        InitResults(False)
    else:
        InitResults(True)
        init_list[3]=True
    
    print(f"{'loading MatchDatabase':40s}",end="")  
    #MatchData :init_list 4
    try:
        data_path='./data/test.pkl'  #test.pkl has 201 datas.
        match_feature=pd.read_pickle(data_path)
    except:
        InitResults(False)
    else:
        InitResults(True)
        init_list[4]=True
        print("DataBase count:",len(match_feature))
    
    print(f"{'setting camera':40s}",end="")
    #cap :init_list 5
    if cv.VideoCapture('/dev/video0').isOpened():
        #建立camera object
        cap=cv.VideoCapture('/dev/video0')
        #setting cap,3:width;4:height;5:FPS;
        cap.set(3,640)
        cap.set(4,480)
        cap.set(5,60)
        #讀取當前frame尺寸640*480
        w=int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        h=int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        InitResults(True)
        print("已偵測到攝影機...")
        print("螢幕尺寸:",w,"x",h,"pi")
        init_list[5]=True
    else:
        InitResults(False)

    InitDivider('loading Model 1/2')
    print(f"{'loading Face Detection Model':40s}",end="")
    #model1 :init_list 6
    try:
        #運行位置
        backend_target_pairs=[cv.dnn.DNN_BACKEND_OPENCV,cv.dnn.DNN_TARGET_CPU]
        #人臉偵測模組
        model_d=yunet.YuNet(modelPath='./model/face_detection/face_detection_yunet_2023mar.onnx',
                    inputSize=[w,h],#camera size
                    confThreshold=0.9,#信心閾值
                    nmsThreshold=0.3,#bbox閾值
                    topK=5000,#top_k(NMS的輸出分數須高於該值才會被判斷為物體)
                    backendId=backend_target_pairs[0],
                    targetId=backend_target_pairs[1])
    except:
        InitResults(False)
    else:
        InitResults(True)
        init_list[6]=True
        print("confTgreshold:",0.9)
        print("nmsTgreshold:",0.3)
        print("topK:",5000)
        
    InitDivider('loading Model 2/2')
    print(f"{'loading Face Recognition Model':40s}",end="")
    #model2 :init_list 7
    try:    #人臉辨識模組
        model_r=cv.FaceRecognizerSF.create(model='./model/face_recognition/face_recognition_sface_2021dec.onnx',
                                          config="",
                                          backend_id=backend_target_pairs[0],
                                          target_id=backend_target_pairs[1])
    except:
        InitResults(False)
    else:
        InitResults(True)
        init_list[7]=True
    
    print(f"{'start thread':40s}",end="")
    #start thread
    try:
        thread2.start()
    except:
        InitResults(False)
    else:
        InitResults(True)
        init_list[8]=True
    

    if not False in init_list:
        InitDivider('complete init',60)
        #start
        print()
        dict1={'state':'','name':'','score':'','r_cd':'','i_cd':'','s_cd':'','HCstate':''}
        for k in dict1.keys():
            print(k+":")
    
        while 1:
            st=time.time()#cycle start
            
            #if infrared detection body:start detection
            if GPIO.input(HC) and i_cd<=0:
                HCstate=1
                r_cd=cd_set[0][0]
                i_cd=sum(cd_set[1])
            #if countdown done :end detection
            elif i_cd<=cd_set[1][1] and HCstate==1:
                HCstate=0
                
            #hasFrame:讀取是否成功,frame:讀取影像
            hasFrame,frame=cap.read()
            if not hasFrame:
                print("")
                print("未偵測到影像")
                InitDivider('endding program',60)                
                print(f"{'closing window':40s}",end="")
                cv.destroyAllWindows()
                InitResults(True)
                print(f"{'closing cam':40s}",end="")
                cap.release()
                InitResults(True)

                InitDivider('end',60)
                break
            
            #flip frame 180 degrees
            frame=cv.flip(frame, 1)
            frame=fv.visualize(frame,mode=2,size=((60,60),(580,420)))
            #check HCstate on :start detection face
            if HCstate:
                
                #add green border
                frame=fv.visualize(frame,mode=2,size=((0,0),(w,h)))
                
                #check face_recognize cooldown <=0:start detection face
                if r_cd<=0:
                    #add threading object for face_detection
                    thread=threading.Thread(target=f,args=(model_d,model_r,frame,q_r,match_feature))
                    
                    #check queue is empty:
                    if q_r.empty():
                        #start threading
                        thread.start()
                        #init ls
                        ls=[0]
                        #get queue
                        ls=q_r.get()
                        if ls[0] != 0:
                            
                            #output=fv.visualize(ls[1],mode=3,string=ls[2])
                            #cv.imshow("face detection",output)
                            #cv.waitKey(1)
                            #time.sleep(1)
                            s_cd=cd_set[2]
                            i_cd=cd_set[1][2]
                        q_r.queue.clear()
                        r_cd=cd_set[0][1]
            if ls !=[0]:
                if ls[0] !=0 and not s_cd<=0:
                    frame=fv.visualize(frame,mode=3,string=ls[2],fps=ct)
                    
            if ls[0] ==1:
                print_check(state="found",name=ls[2],score=ls[3],r_cd=round(r_cd,1),i_cd=round(i_cd,1),HCstate=HCstate,s_cd=round(s_cd))
            else:
                print_check(state="",name="",score="",r_cd=round(r_cd,1),i_cd=round(i_cd,1),HCstate=HCstate,s_cd=round(s_cd))
                
            cv.imshow('face detection',frame)
            if cv.waitKey(1)&0xff == ord("q"):
                print("")
                InitDivider('endding program',60)                
                print(f"{'closing window':40s}",end="")
                cv.destroyAllWindows()
                InitResults(True)
                print(f"{'closing cam':40s}",end="")
                cap.release()
                InitResults(True)

                InitDivider('end',60)
                break
            
            if not q_btn.empty():
                butt=q_btn.get()[0]
                if butt==2:
                    print("")
                    InitDivider('endding program',60)
                    print(f"{'closing window':40s}",end="")
                    cv.destroyAllWindows()
                    InitResults(True)
                    print(f"{'closing cam':40s}",end="")
                    cap.release()
                    InitResults(True)

                    InitDivider('end',60)
                    break
                q_btn.queue.clear()
            ct=time.time()-st
            if r_cd >0 and HCstate:
                r_cd-=ct
            if i_cd >0:
                i_cd-=ct
            if s_cd >0:
                s_cd-=ct
           #print("\r","r_cd:",round(r_cd,1),"i_cd:",round(i_cd,1),'HCstate:',HCstate,end=" ")
    else:InitDivider('complete init',60)
if __name__ == "__main__":
    main()

            
