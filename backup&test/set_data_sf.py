from function import yunet
import numpy as np
import os
import cv2 as cv
import pandas as pd
import pickle

input_model="face_recognition_sface_2021dec.onnx"
onnx_path = os.path.join("./model/face_recognition/",input_model)

backend_target_pairs=[cv.dnn.DNN_BACKEND_OPENCV,cv.dnn.DNN_TARGET_CPU]
model_d=yunet.YuNet(modelPath='./model/face_detection/face_detection_yunet_2023mar.onnx',
            inputSize=[320,320],
            confThreshold=0.9,
            nmsThreshold=0.3,
            topK=5000,
            backendId=backend_target_pairs[0],
            targetId=backend_target_pairs[1])
model_r=cv.FaceRecognizerSF.create(model=onnx_path,
                                         config="",
                                         backend_id=backend_target_pairs[0],
                                         target_id=backend_target_pairs[1]
                                         )
matchdata={}
save_path='./data/feature.pkl'
# save_path='./data/feature.xlsx'

def visualize(image, results, box_color=(0, 255, 0), text_color=(0, 0, 255),fps=None,norm=None,Cosine_similarity=None):
    output = image.copy()
    landmark_color = [#BGR
        (255,   0,   0), # right eye 藍
        (  0,   0, 255), # left eye 紅
        (  0, 255,   0), # nose tip 綠
        (255,   0, 255), # right mouth corner 粉
        (  0, 255, 255)  # left mouth corner  黃
    ]

    if fps is not None:
        cv.putText(output, 'FPS: {:.2f}'.format(fps), (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, text_color)
    

    for det in results:
        bbox = det[0:4].astype(np.int32)
        cv.rectangle(output, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), box_color, 2)
        if norm is not None:
            cv.putText(output, 'norm: {:.2f}'.format(norm), (bbox[0], bbox[1]+30), cv.FONT_HERSHEY_DUPLEX, 0.5, text_color)
        if Cosine_similarity is not None:
            cv.putText(output, 'Cs: {:.2f}'.format(Cosine_similarity), (bbox[0], bbox[1]+30), cv.FONT_HERSHEY_DUPLEX, 0.5, text_color)
        conf = det[-1]
        cv.putText(output, '{:.4f}'.format(conf), (bbox[0], bbox[1]+12), cv.FONT_HERSHEY_DUPLEX, 0.5, text_color)

        landmarks = det[4:14].astype(np.int32).reshape((5,2))
        #print(landmarks)
        for idx, landmark in enumerate(landmarks):
            cv.circle(output, landmark, 2, landmark_color[idx], 2)#圖原,中心,半徑,顏色,粗細

    return output

def affineMatrix(lmks,image,scale=2):
    lmks = lmks[0][4:14].astype(np.int32).reshape((5,2))
    nose=np.array(lmks[2],dtype=np.float32)
    left_eye=np.array(lmks[1],dtype=np.float32)
    right_eye=np.array(lmks[0],dtype=np.float32)
    eye_width=left_eye-right_eye
    
    
    angle=np.arctan2(eye_width[1],eye_width[0])
    center=nose
    alpha=np.cos(angle)
    beta=np.sin(angle)
    w=np.sqrt(np.sum(eye_width**2))*scale
    m = [[alpha, beta, -alpha * center[0] - beta * center[1] + w * 0.5],[-beta, alpha, beta * center[0] - alpha * center[1] + w * 0.6]]
    image=cv.warpAffine(image,np.array(m),(int(w),int(w)))#affine matrix,target size
    # image=cv.resize(image,(112,112))
    return image

if __name__=="__main__":
    if "feature.pkl" in os.listdir('./data'):
        matchdata=pd.read_pickle(save_path)
        
    if cv.VideoCapture(0).isOpened():
        name=""
        print("已偵測到攝影設備")
        cap=cv.VideoCapture(0)
        
        tm=cv.TickMeter()
        w=int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        h=int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        print("攝影機規格:",w,"x",h,"pi")
        model_d.setInputSize([w,h])
        
        while 1:
            hasFrame,frame=cap.read()
            if not hasFrame:
                print("未讀取到影像")
                cv.destroyAllWindows()
                break
            frame=cv.flip(frame,1)
            image=frame.copy()
            
            tm.start()
            results=model_d.infer(frame)
            
            if results.shape[1]==15:
                frame=visualize(frame, results)
            tm.stop()
            cv.imshow('stream',frame)
            tm.reset()
            if cv.waitKey(1)&0xff ==ord('s'):
                    name=input("請輸入名稱:")
                    input_r=model_r.alignCrop(image, results)
                    feature_r=model_r.feature(input_r)
                    matchdata[name]=feature_r
                    pd.to_pickle(matchdata,save_path)
                    # pickle.dump(matchdata,save_path)

                    cv.destroyAllWindows()
                    print("檔案建立完成")
                    break
            if cv.waitKey(1)&0xff ==ord('q'):
                cv.destroyAllWindows()
                print("取消建立")
                break