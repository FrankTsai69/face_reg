import numpy as np
import cv2 as cv 
import pandas as pd
import os
from function import yunet
from function import face_visualize as fv
from function import face_feature as ff
import pickle
#偵測圖檔位置
target_path="./media/target"
#運行位置
backend_target_pairs=[cv.dnn.DNN_BACKEND_OPENCV,cv.dnn.DNN_TARGET_CPU]
#人臉偵測模組
model_d=yunet.YuNet(modelPath='./model/face_detection/face_detection_yunet_2023mar.onnx',
            inputSize=[320,320],
            confThreshold=0.9,#信心閾值
            nmsThreshold=0.3,#bbox閾值
            topK=5000,#top_k(NMS的輸出分數須高於該值才會被判斷為物體)
            backendId=backend_target_pairs[0],
            targetId=backend_target_pairs[1])

#人臉辨識模組
model_r=cv.FaceRecognizerSF.create(model='./model/face_recognition/face_recognition_sface_2021dec.onnx',
                                  config="",
                                  backend_id=backend_target_pairs[0],
                                  target_id=backend_target_pairs[1])

data_path='./data/feature.pkl'
match_meth=1
#輸入來源
input=0


     
    
if __name__=='__main__':
    # match_feature=pd.read_pickle(data_path)
    with open(data_path,'rb') as t:
        match_feature=pickle.load(t)
    #輸入圖片
    if input==1:
        image_list=os.listdir(target_path)
        if image_list !=[]:
            for i in image_list:
                image=cv.imread(os.path.join(target_path,i))
                h,w,_=image.shape#高、寬、像素通道
                # image=cv.resize(image,(h//10,h//10))
                # h,w,_=image.shape#高、寬、像素通道
                #人臉偵測
                model_d.setInputSize([w,h])
                results=model_d.infer(image)
                
                if results.shape[1] !=15:
                    print(i[:-4],"未偵測到人臉")
                    continue
                #特徵建立
                feature=ff.feature(model_r,image,results)

                #比較
                name=ff.match(model_r,feature,match_feature,match_meth,0.7)
                final_image=fv.visualize(image, results,name=name,score_type=match_meth)   
                cv.imshow('image',final_image)
                cv.waitKey(0)
                cv.destroyAllWindows()
        elif image_list==[]:
            print('目標資料夾為空')
    #輸入串流並確認是否接收到設備
    elif input==0 and cv.VideoCapture(0).isOpened():
        name={}
        match_meth=None
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
        
        while 1:
            name={}
            #hasFrame:讀取是否成功,frame:讀取影像
            hasFrame,frame=cap.read()           
            if not hasFrame:
                print("未偵測到影像")
                cv.destroyAllWindows()
                break
            frame=cv.flip(frame, 1)
            tm.start()
            #人臉偵測
            results=model_d.infer(frame)#results是一組tuple
            for i in range(len(results)):
                name[i+1]={"results":results[i-1],"feature":[],"name":"none","score":0}
            if results.shape[1] ==15:
                #特徵值擷取
                name=ff.feature(model_r,frame,name)
                match_meth=1
                #比較
                name=ff.match(model_r,name,match_feature,match_meth,0.9)
                time=0
            tm.stop()
            time+=tm.getTimeSec()
            # print(time)
            image=fv.visualize(frame,name=name,score_type=match_meth,fps=tm.getFPS(),mode=1)
            
            #新建視窗並輸出視覺化後的影像
            # thread2.start()
            cv.imshow('face detection',image)
            # time+=tm.getAvgTimeSec()
            # print(round(time,1))
            tm.reset()
            #當偵測到數據庫的人物時輸出
            # if name !="unknown" and name !="":
            #     cv.waitKey(0)
            #     cv.destroyAllWindows()
            #     return results,frame
            if cv.waitKey(1)&0xff == ord("q"):
                cv.destroyAllWindows()
                break
