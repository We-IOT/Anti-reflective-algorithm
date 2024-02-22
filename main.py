import cv2
import numpy as np

# 定义ArUco码的参数
#aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
ARUCO_DICT = {
     "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
     "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
     "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
     "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
     "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
     "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
     "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
     "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
     "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
     "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
     "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
     "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
     "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
     "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
     "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
     "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
     "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
     "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
     "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
     "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
     "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
 }
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)


# 设置检测器参数  
params = cv2.aruco.DetectorParameters_create()  
# params.adaptiveThreshWinSizeMin = 3  # 调整为更小的窗口大小  
# params.adaptiveThreshWinSizeMax = 9  # 调整为更大的窗口大小  
# params.adaptiveThreshConstant = 1    # 常数调整  
#params.minMarkerPerimeterRate = 5             # 最小标记大小  
# whatparams.maxMarkerPerimeterRate = 50            # 最大标记大小  
aruco_params = params

# 打开视频文件
cap = cv2.VideoCapture('test.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # 转换为灰度图像  
    image=frame[300:1400,1000:2000]
    #image=frame
    # 转换为HSV色彩空间（可选）  
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  
    
    # 分离HSV通道  
    h, s, v = cv2.split(hsv_image)  
    
    # 创建CLAHE对象并应用于V通道  
    clahe = cv2.createCLAHE(clipLimit=32.0, tileGridSize=(8,8))  
    v_clahe = clahe.apply(v)  
    
    # 合并处理后的HSV通道  
    hsv_image = cv2.merge([h, s, v_clahe])  
    
    # 转换回BGR色彩空间（如果之前转换过）  
    image_clahe = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)


    frame = image_clahe
    # 识别ArUco码
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(frame , aruco_dict, parameters=aruco_params)
    print(corners, ids)
    # 如果识别到了ArUco码,则在视频上绘制边框
    if len(corners) > 0:
        # flatten the ArUco IDs list
        print(ids)
        ids = ids.flatten()
        # loop over the detected ArUCo corners
        for (markerCorner, markerID) in zip(corners, ids):
            print(markerCorner)
            # extract the marker corners (which are always returned in
            # top-left, top-right, bottom-right, and bottom-left order)
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners
            # convert each of the (x, y)-coordinate pairs to integers
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))
            
            
            # draw the bounding box of the ArUCo detection
            cv2.line(frame, topLeft, topRight, (0, 255, 0), 4)
            cv2.line(frame, topRight, bottomRight, (0, 255, 0), 4)
            cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 4)
            cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 4)
            # compute and draw the center (x, y)-coordinates of the ArUco
            # marker
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(frame, (cX, cY), 8, (0, 0, 255), -1)
            # draw the ArUco marker ID on the image
            cv2.putText(frame, str(markerID),
                (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 4)
            print("[INFO] ArUco marker ID: {}".format(markerID))
        # 显示视频帧
    cv2.imshow('frame', frame)

    # 按下q键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite('aruco_marker.png', frame)
        break

cap.release()
cv2.destroyAllWindows()
