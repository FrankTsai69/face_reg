import numpy as np
import os
import cv2 as cv
import pandas as pd
import pickle

matchdata={}
save_path='./data/test.pkl'
if "feature.pkl" in os.listdir('./data'):
        matchdata=pd.read_pickle(save_path)
feature=np.ones(matchdata["Frank"].shape,dtype=np.float32)

print(matchdata["Frank"])
print("")
print(matchdata[1])
for i in range(99):
	matchdata[str(i)]=feature
	matchdata[i]=feature
	pd.to_pickle(matchdata,'./data/test.pkl')
	print("檔案建立完成",i)
