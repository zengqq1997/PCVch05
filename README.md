# PCVch05

记录学习Python Computer Vision的过程

第六次记录

## python-opencv进行相机标定

### 前言

如今大量廉价的摄像机导致了很多照片畸变。两个主要的畸变是径向畸变和切向畸变。

畸变（distortion）是对直线投影（rectilinear projection）的一种偏移。简单来说直线投影是场景内的一条直线投影到图片上也保持为一条直线。畸变简单来说就是一条直线投影到图片上不能保持为一条直线了，这是一种光学畸变（optical aberration）,可能由于摄像机镜头的原因

 由于径向畸变，直线会变弯。距离图片中心越远，它的影响越大。如下面这张图片，棋盘格中被红线标记的边缘。你会发现棋盘格的边缘并不与直红线重合，也就是说棋盘格边缘已经不平行了。

![img](https://github.com/zengqq1997/PCVch05/blob/master/Inked05_LI.jpg)

那么接下来我们来进行相机标定

### 基础

**相机标定的意义**：在机器视觉领域，相机的标定是一个关键的环节，它决定了机器视觉系统能否有效的定位，能否有效的计算目标物。相机的标定基本上可以分为两种，第一种是相机的自标定；第二种是依赖于标定参照物的标定方法。前者是相机拍摄周围物体，通过数字图像处理的方法和相关的几何计算得到相机参数，但是这种方法标定的结果误差较大，不适合于高精度应用场合。后者是通过标定参照物，由相机成像，并通过数字图像处理的方法，以及后期的空间算术运算计算相机的内参和外参。这种方法标定的精度高，适用于对精度要求高的应用场合。

**坐标转换基础**：在视觉测量中，需要进行的一个重要预备工作是定义四个坐标系的意义，即 摄像机坐标系、 图像物理坐标系、 像素坐标系 和 世界坐标系（参考坐标系） 。

### 实例

**相机标定步骤：**

> 1、打印一张棋盘格，把它放在一个平面上，作为标定物。 
> 2、通过调整标定物或摄像机的方向，为标定物拍摄一些不同方向的照片。 
> 3、从照片中提取棋盘格角点。 
> 4、标定结果、相机的内参数矩阵、畸变系数、旋转矩阵和平移向量。 
> 5、去畸变。 
> 6、求反投影误差。

#### 1.准备工作

如上步骤所述，我通过小米手机拍摄十二张不同角度的测试模板图片来进行相机标定。而且我这里选择的是10×6的棋盘格，里面有9×5的内格子。例图如下所示

![img](https://github.com/zengqq1997/PCVch05/blob/master/01.jpg)

#### 2.检测棋盘格角点

为了找到棋盘格模板，我们使用openCV中的函数cv2.findChessboardCorners()。我们也需要告诉函数我们使用的模板是什么规格的，例如10×6的棋盘格或者5*5棋盘格等，建议使用x方向和y方向个数不相等的棋盘格模板。因为我们使用的是有9×5的内格子，所以我们在设置w和h时分别填写9和5。函数返回每一一个角点，如果匹配到了模式，它将返回是True。这些角点将按一定顺序标注出来（从左到右，从上到下） 。

找到角点后，我们可以使用cv2.cornerSubPix()可以得到更为准确的角点像素坐标。我们也可以使用cv2.drawChessboardCorners()将角点绘制到图像上显示。实验结果如下图

![img](https://github.com/zengqq1997/PCVch05/blob/master/conimg5.jpg)

#### 3.标定

通过上面的步骤，我们得到了用于标定的三维点和与其对应的图像上的二维点对。我们使用cv2.calibrateCamera()进行标定，这个函数会返回标定结果、相机的内参数矩阵、畸变系数、旋转矩阵和平移向量。计算的参数如下

**分别是相机内参数矩阵、畸变系数、旋转向量、平移向量**

![img](https://github.com/zengqq1997/PCVch05/blob/master/num.jpg)

#### 4.去畸变

第三步我们已经得到了相机内参和畸变系数，在将图像去畸变之前，我们还可以使用cv.getOptimalNewCameraMatrix()优化内参数和畸变系数，通过设定自由自由比例因子alpha。当alpha设为0的时候，将会返回一个剪裁过的将去畸变后不想要的像素去掉的内参数和畸变系数；当alpha设为1的时候，将会返回一个包含额外黑色像素点的内参数和畸变系数，，并返回一个ROI用于将其剪裁掉。如下图的新的内参数矩阵

![img](https://github.com/zengqq1997/PCVch05/blob/master/num2.jpg)

然后我们就可以使用新得到的内参数矩阵和畸变系数对图像进行去畸变了。有两种方法进行去畸变：

-   使用cv2.undistort()
- 使用remmaping

那么这里使用第一种方式来去畸变。得到结果如下所示：

![img](https://github.com/zengqq1997/PCVch05/blob/master/calibresult.jpg)

#### 5.求反投影误差

通过反投影误差，我们可以来评估结果的好坏。越接近0，说明结果越理想。通过之前计算的内参数矩阵、畸变系数、旋转矩阵和平移向量，使用cv2.projectPoints()计算三维点到二维图像的投影，然后计算反投影得到的点与图像上检测到的点的误差，最后计算一个对于所有标定图像的平均误差，这个值就是反投影误差。

### 实验代码

```python
#coding:utf-8
import cv2
import numpy as np
import glob

# 找棋盘格角点
# 阈值
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#棋盘格模板规格
w = 9
h = 5
# 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
objp = np.zeros((w*h,3), np.float32)
objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
# 储存棋盘格角点的世界坐标和图像坐标对
objpoints = [] # 在世界坐标系中的三维点
imgpoints = [] # 在图像平面的二维点

images = glob.glob('C:/Users/ZQQ/Desktop/advanced/study/computervision/images/ch05/*.jpg')
i=0;
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # 找到棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, (w,h),None)
    # 如果找到足够点对，将其存储起来
    if ret == True:
        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        objpoints.append(objp)
        imgpoints.append(corners)
        # 将角点在图像上显示
        cv2.drawChessboardCorners(img, (w,h), corners, ret)
        cv2.imshow('findCorners',img)
        cv2.waitKey(1)
        i+=1;
        cv2.imwrite('conimg'+str(i)+'.jpg', img)

# 标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("ret:", ret)
print("mtx:\n", mtx) # 内参数矩阵
print("dist:\n", dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
print("rvecs:\n", rvecs)  # 旋转向量  # 外参数
print("tvecs:\n", tvecs ) # 平移向量  # 外参数

print("-----------------------------------------------------")
# 去畸变
img2 = cv2.imread(images[2])
h,  w = img2.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),0,(w,h)) # 自由比例参数
dst = cv2.undistort(img2, mtx, dist, None, newcameramtx)
print("newcameramtx:\n",newcameramtx)
# 根据前面ROI区域裁剪图片
#x,y,w,h = roi
#dst = dst[y:y+h, x:x+w]
cv2.imshow('findCorners', dst)
cv2.waitKey(1)
cv2.imwrite('calibresult.png',dst)

# 反投影误差
total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    total_error += error
print("total error: ", total_error/len(objpoints))
```

### 实验过程遇到的问题

![img](https://github.com/zengqq1997/PCVch05/blob/master/error.jpg)

**解决**：在python2上运行会报错，将其换到python3就可以了。以及注意棋盘格的內格子是否设置对了。

### 小结

本节了解导致相机失真、扭曲的内因与外因，并实现了找到图像的这些畸变参数，并消除畸变。

