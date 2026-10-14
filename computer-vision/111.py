import cv2
import numpy as np
import glob

# 找棋盘格角点
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
w = 8   
h = 6
objp = np.zeros((w*h, 3), np.float32)
objp[:,:2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
objpoints = []
imgpoints = []

# 读取标定图（路径已确认正确）
images = glob.glob(r"E:\SJ\photo\*.jpg")
for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"警告：无法读取图片 {fname}，请检查是否损坏")
        continue
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)  # 模糊降噪
    
    # 优化角点检测参数
    ret, corners = cv2.findChessboardCorners(
        gray, (w, h), 
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    
    if ret == True:
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners)
        cv2.drawChessboardCorners(img, (w, h), corners, ret)
        cv2.putText(img, f"Detected {w}x{h} corners", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.namedWindow('findCorners', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('findCorners', 800, 600)
        cv2.imshow('findCorners', img)
        cv2.waitKey(1000)
cv2.destroyAllWindows()

# 检查有效标定图
if len(objpoints) == 0:
    print("错误：仍未找到有效棋盘格角点！")
    print("最后确认：")
    print(f"1. 内角点参数：w={w}, h={h}（已根据图片修正）；")
    print("2. 标定图（1.jpg、2.jpg）是否在 E:\SJ\photo 文件夹且为.jpg格式；")
    print("3. 图片是否清晰、棋盘格完整无反光。")
    exit()

# 相机标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("="*50)
print("标定成功！输出参数：")
print("ret:", ret)
print("内参数矩阵:\n", mtx)
print("畸变系数:\n", dist)
print("旋转向量:\n", rvecs)
print("平移向量:\n", tvecs)
print("="*50)
img2 = cv2.imread(r"E:\SJ\photo\1.jpg")
if img2 is None:
    print("错误：无法读取3.jpg，请检查是否在 E:\SJ\photo 文件夹！")
    exit()

h_img, w_img = img2.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w_img, h_img), 0, (w_img, h_img))
dst = cv2.undistort(img2, mtx, dist, None, newcameramtx)
x, y, roi_w, roi_h = roi
dst = dst[y:y+roi_h, x:x+roi_w]
cv2.imwrite(r"E:\SJ\photo\calibresult.jpg", dst)
print("去畸变图片已保存至：E:\SJ\photo\calibresult.jpg")

# 反投影误差
total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    total_error += error
print("反投影误差（越接近0精度越高）:", total_error / len(objpoints))