import numpy as np
import cv2 as cv
from sklearn.preprocessing import normalize
import pandas as pd
import os
import onnxruntime as rt
#feature
#特徵擷取模型
input_model=[
    "arcface_r100_v1.onnx",
    "webface_r50.onnx",
    "face_recognition_sface_2021dec.onnx",
    "glintasia_r50_pfc.onnx",
    "webface_r50_pfc.onnx"
    ]

class feature:
    def __init__(self,model=2,dis_threshold=0.4):
        self._onnx_path = os.path.join("./model/face_recognition/",input_model[model])
        self._threshold=dis_threshold
        self.data_embedding=pd.read_pickle("./data/feature.pkl")
        #模型部屬
        self.extractor=rt.InferenceSession(self._onnx_path)
    def feature_extraction(self,image):        
        #輸入模型
        
        final_embedding=[]
        # distance_di={}
        image=cv.resize(image,(112,112))
        t_aligned=np.transpose(image,(2,0,1))
        inputs=t_aligned.astype(np.float32)
        
        #在input_blob的維度0增加資料inputs
        input_blob=np.expand_dims(inputs, axis=0)
        
        #取得第一個輸入資訊的節點名稱
        first_input_name=self.extractor.get_inputs()[0].name
        #取得第一個輸出資訊的節點名稱
        first_output_name=self.extractor.get_outputs()[0].name
        
        #啟動模型
        predict=self.extractor.run([first_output_name],{first_input_name:input_blob})[0]
        #重組特徵值矩陣
        final_embedding.append(normalize(predict).flatten())
        di={}
        for k in self.data_embedding.keys():
            di[k]=(round(sum((final_embedding[0]-self.data_embedding[k])**2)**0.5,2))
        for k in di.keys():
            print(k,":",di[k])
            if di[k]<self._threshold and di[k]==min(di.values()):
                return k
        
        return "unknown"
    