import numpy as np
import cv2 as cv
import pandas as pd
import os
import operator
#feature
#特徵擷取模型
input_model="face_recognition_sface_2021dec.onnx"
def feature(model,target,name):
    
    if len(name)==1:
        result=name[1]["results"][:-1]
        target_input=model.alignCrop(target,result)
        name[1]["feature"]=model.feature(target_input)
        return name
    elif len(name)>1:
        for i in name.keys():
            result=name[i]["results"][:-1]
            target_input=model.alignCrop(target,result)
            name[i]["feature"]=model.feature(target_input)
        return name
    return name
     
def match(model,name,match_feature,match_type,threshold):
    score_dict={}

    if len(name)==1:
        
        if match_type==1:    
            for k in match_feature.keys():
                f=match_feature[k]
                score=round(model.match(name[1]["feature"],f,match_type),2)
                score_dict[k]=score
            
            winner=min(score_dict.items(),key=operator.itemgetter(1))[0]
            score_winner=score_dict[winner]
            if score_winner<=threshold and winner !='none':
                name[1]["name"]=winner
                name[1]["score"]=score_winner
                return name
        elif match_type==0:
            for k in match_feature.keys():
                f=match_feature[k]
                score=round(model.match(name[1]["feature"],f,match_type),2)
                score_dict[k]=score
                
            winner=max(score_dict.item(),key=operator.itemgetter(1))[0]
            score_winner=score[winner]
            if score_winner>=threshold and winner !='none':
                name[1]["name"]=winner[1]
                name[1]["score"]=score_winner
                return name
        name[1]["name"]="unknown"
        name[1]["score"]=-1
        return name
    elif len(name)>1:
        
        for i in name.keys():
            for k in match_feature.keys():
                f=match_feature[k]
                score=round(model.match(name[i]["feature"],f,match_type),2)
                score_dict[k]=score
            if match_type==1:
                winner=min(score_dict.items(),key=operator.itemgetter(1))[0]
                score_winner=score_dict[winner]
                if score_winner<=threshold:
                    name[i]["name"]=winner
                    name[i]["score"]=score_winner
                    continue
            elif match_type==0:
                winner=max(score_dict.item(),key=operator.itemgetter(1))[0]
                score_winner=score[winner]
                if score_winner>=threshold:
                   name[i]["name"]=winner
                   name[i]["score"]=score_winner
                   continue
            name[i]["name"]="unknown"
            name[i]["score"]=score_winner
        return name
        
# if __name__ == "__main__":
