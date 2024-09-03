# 專案:結合邊緣運算裝置的人臉辨識系統
## 今天進度:
  1. 測試超音波偵測與紅外線偵測的功能
     * 優缺點:
        * 超音波會有干擾導致迴圈中斷的問題；但可以依照距離判斷物體遠近
        * 紅外線因為無法控制中斷時間，所以會有持續時間結束後2.5秒不能進行辨識的問題；不會有迴圈中斷的問題
      * 結論:
         * 使用紅外線
  2. 進行視覺化函數(face_visulalize)的功能擴充
     1. 將testdemo中的視覺化功能移入該函數
     2. 流程更新
     3. 新增變數mode、string
        * mode:
          * 0:單人偵測使用，可顯示名稱、分數、bbox
          * 1:多人偵測使用，可顯示名稱、分數、bbox、conf值
          * 2:為圖片增加綠色邊框
          * 3:顯示文字在中間偏下面，文字為string所輸入的內容
        * string:
          * mode:3使用
  3. 載入清單更新
  4. 數值顯示更新(需要再改進)
  5. 新增紅外線偵測用燈
  6. 現在攝影機初始化會將尺寸與偵數固定
  7. 現在結束時會將攝影機關閉才結束迴圈，之前關閉程式會有一堆陣列出現就是因為沒有先關攝影機
## 代辦清單:
  1. 數值顯示找到更好的方式
  2. 新增簽到資料輸出
  3. 新增關機按鈕
  4. 