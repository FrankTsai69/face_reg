# 結合邊緣運算裝置的人臉辨識系統
## 簡介
能在樹梅派4上運作的人臉辨識系統
## 設備
* 人臉偵測模型:YuNet(Source:[OpenCV_Zoo](<https://github.com/opencv/opencv_zoo>))
* 人臉辨識模型:Sface(Source:[OpenCV_Zoo](<https://github.com/opencv/opencv_zoo>))
* 樹梅派4
* 按鈕*1(pin 23)
* 人體紅外線偵測模組(HC-SR501)(pin 18)
* led燈(pin 26)
## 使用方法
將檔案匯入樹梅派4後，利用crontab指令將auto_script中的face_reg_btn.sh放入排程並設定為開機時啟動，重新啟動後長按按鈕2秒啟動主程式
