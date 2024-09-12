import cv2 as cv
import numpy as np
#視覺化函數，在圖片上新增bbox、landmark、fps...
def visualize_border(image,color=2,size=1):
    output = image.copy()
    color_list=[#BGR
        (255,   0,   0), #0  blue       
        (  0,   0, 255), #1  red     
        (  0, 255,   0), #2  green   
        (255, 255,   0), #3  cyan    
        (255,   0, 255), #4  pink    
        (  0, 255, 255), #5  yellow  
        (255, 255, 255), #6  white   
        (  0,   0,   0)  #7  black   
    ]
    cv.rectangle(output, (size[0][0], size[0][1]), (size[1][0], size[1][1]), color_list[color], 2)
    return output
    
def visualize_string(
          image,                    #image source
          string,                   #string 
          coordinate,               #string coordinate
          string_font=2,            #string font
          string_scale=0.5,           #string size
          string_color=7,           #string color;default=7(black)
          align='left',   
          background=False,
          ):      
    color_list=[#BGR
        (255,   0,   0), #0  blue       
        (  0,   0, 255), #1  red     
        (  0, 255,   0), #2  green   
        (255, 255,   0), #3  cyan    
        (255,   0, 255), #4  pink    
        (  0, 255, 255), #5  yellow  
        (255, 255, 255), #6  white   
        (  0,   0,   0)  #7  black   
    ]
    font_list=[
        cv.FONT_HERSHEY_SIMPLEX,        #0 normal size sans-serif font
        cv.FONT_HERSHEY_PLAIN,          #1 small size sans-serif font
        cv.FONT_HERSHEY_DUPLEX,         #2 normal size sans-serif font (more complex than FONT_HERSHEY_SIMPLEX)
        cv.FONT_HERSHEY_COMPLEX,        #3 normal size serif font
        cv.FONT_HERSHEY_TRIPLEX,        #4 normal size serif font (more complex than FONT_HERSHEY_COMPLEX)
        cv.FONT_HERSHEY_COMPLEX_SMALL,  #5 smaller version of FONT_HERSHEY_COMPLEX
        cv.FONT_HERSHEY_SCRIPT_SIMPLEX, #6 hand-writing style font
        cv.FONT_HERSHEY_SCRIPT_COMPLEX  #7 more complex variant of FONT_HERSHEY_SCRIPT_SIMPLEX
    ]
    output = image.copy()
    font=font_list[string_font]
    color=color_list[string_color]

    
    #get text size
    text_w,text_h=cv.getTextSize(string,font,string_scale,1)[0]
    match align:
        case 'left':
            x,y=int(coordinate[0]),int(coordinate[1])
        case 'center':
            x,y=int(coordinate[0]-text_w/2),int(coordinate[1])
        case 'right':
            x,y=int(coordinate[0]-text_w),int(coordinate[1])
    if background:
        bg_x=text_w+10
        bg_y=text_h+10
        size=output[int(y-bg_y//2-2):int(y+bg_y//2-2),x+1:(x+bg_x+1)]
        bg=np.zeros((size.shape[0],size.shape[1],3),dtype=np.uint8)
        bg[:,:,:]=255
        img_add=cv.addWeighted(size,0.7,bg,0.3,100)
        output[int(y-bg_y//2-2):int(y+bg_y//2-2),x+1:(x+bg_x+1)]=img_add
        
    cv.putText(output, string, (x,y),font,string_scale,color)
    
    return output
