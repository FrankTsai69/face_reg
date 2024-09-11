import cv2 as cv
import numpy as np
#視覺化函數，在圖片上新增bbox、landmark、fps...

def visualize(image,
              name=None,
              box_color=(0, 255, 0),
              mode=0,
              text_color=(0, 0, 255),
              fps=None,score_type=None,
              size=((0,0),(0,0)),
              string=""):
                  
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
    if mode==0:
        output = image.copy()
        box_color=(0, 255, 0)
        text_color=(0, 0, 255)
        det=name[1]["results"]
        #bbox框
        bbox = det[0:4].astype(np.int32)
        cv.rectangle(output, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), box_color, 2)
        if "name" in name[1].keys():
            #名稱
            cv.putText(output, '{}'.format(name[1]["name"]), (bbox[0], bbox[1]-10), cv.FONT_HERSHEY_DUPLEX, 0.5, text_color)
        # 分數
        if score_type==1:
            cv.putText(output, 'norm: {:.2f}'.format(name[i]["score"]), (bbox[0], bbox[1]+30), cv.FONT_HERSHEY_DUPLEX, 0.5, text_color)
        elif score_type==0:
            cv.putText(output, 'Cs: {:.2f}'.format(name[i]["score"]), (bbox[0], bbox[1]+30), cv.FONT_HERSHEY_DUPLEX, 0.5, text_color)
    elif mode==1:
        for i in name.keys():
            det=name[i]["results"]
            #bbox框
            bbox = det[0:4].astype(np.int32)
            cv.rectangle(output, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), box_color, 2)
            #名稱
            cv.putText(output, '{}'.format(name[i]["name"]), (bbox[0], bbox[1]-10), cv.FONT_HERSHEY_DUPLEX, 0.5, text_color)
            # 分數
            if score_type==1:
                cv.putText(output, 'norm: {:.2f}'.format(name[i]["score"]), (bbox[0], bbox[1]+30), cv.FONT_HERSHEY_DUPLEX, 0.5, text_color)
            elif score_type==0:
                cv.putText(output, 'Cs: {:.2f}'.format(name[i]["score"]), (bbox[0], bbox[1]+30), cv.FONT_HERSHEY_DUPLEX, 0.5, text_color)
                
            conf = det[-1]
            cv.putText(output, '{:.4f}'.format(conf), (bbox[0], bbox[1]+12), cv.FONT_HERSHEY_DUPLEX, 0.5, text_color)

            landmarks = det[4:14].astype(np.int32).reshape((5,2))
            #print(landmarks)
            for idx, landmark in enumerate(landmarks):
                cv.circle(output, landmark, 2, landmark_color[idx], 2)#圖原,中心,半徑,顏色,粗細
    elif mode==2:
        cv.rectangle(output, (size[0][0], size[0][1]), (size[1][0], size[1][1]), box_color, 2)
    elif mode==3:
        if len(string)%2==0:
            num=int(len(string)/2)*7
        else:
            num=int((len(string)+1)/2)*7
        cv.putText(output, '{}'.format(string), (320-num,360), cv.FONT_HERSHEY_DUPLEX, 0.5, text_color)
        output=cv.addWeighted(output,1)
    return output
