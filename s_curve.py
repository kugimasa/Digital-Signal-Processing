import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2gray
from numpy import uint8, float32, float64, log, pi, sin, cos, abs, sqrt
from scipy.ndimage.filters import convolve

def realtime_graph(x, y):
    line, = plt.plot(x, y,label="S Curve") # (x,y)のプロット
    line.set_ydata(y)   # y値を更新
    plt.title("S Curve")  # グラフタイトル
    plt.xlabel("x")     # x軸ラベル
    plt.ylabel("y")     # y軸ラベル
    plt.legend()        # 凡例表示
    plt.grid()          # グリッド表示
    plt.xlim([0,255])    # x軸範囲
    plt.ylim([0,255])    # y軸範囲
    plt.draw()          # グラフの描画
    plt.pause(0.01)     # 更新時間間隔
    plt.clf()

def myfunc(i):
    pass # do nothing

def S_curve(x):
    y = (np.sin(np.pi * (x/255 - s/255)) + 1)/2 * 255
    return y

#filter
Px = np.array([[-1,-1,-1],
               [ 0, 0, 0],
               [ 1, 1, 1]])
Py = np.array([[-1, 0, 1],
               [-1, 0, 1],
               [-1, 0, 1]])
Sx = np.array([[-1,-2,-1],
               [ 0, 0, 0],
               [ 1, 2, 1]])
Sy = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]])

def myfilter(x,y,im):
    Ix =  convolve(im,x)
    Iy =  convolve(im,y)
    M = sqrt(Ix**2 + Iy**2)
    return M


cv2.namedWindow('title') # create win with win name
#RGB trackbar
cv2.createTrackbar('R','title',100,100,myfunc)
cv2.createTrackbar('G','title',100,100,myfunc)
cv2.createTrackbar('B','title',100,100,myfunc)
cv2.createTrackbar('grayscale','title',0,1,myfunc)
cv2.createTrackbar('Sobel Filter:on/off','title',0,1,myfunc)
cv2.createTrackbar('Prewitt Filter:on/off','title',0,1,myfunc)
cv2.createTrackbar('s_curve','title',127,255,myfunc)
cv2.createTrackbar('S Curve:on/off','title',0,1,myfunc)



cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


while(True):
    k = cv2.waitKey(1)
    if k == ord('q') or k == 27:
        break
    
    ret, frame = cap.read()
    if not ret: continue

    #frame resize
    #frame = cv2.resize(frame,(int(frame.shape[1]/3), int(frame.shape[0]/3)))

    r = cv2.getTrackbarPos('R','title')
    g = cv2.getTrackbarPos('G','title')
    b = cv2.getTrackbarPos('B','title')

    #rbg
    R = (frame[:,:,2]/255)*(r/100)
    G = (frame[:,:,1]/255)*(g/100)
    B = (frame[:,:,0]/255)*(b/100)
    frame[:,:,2] = R*255
    frame[:,:,1] = G*255
    frame[:,:,0] = B*255

    #grayscale
    gray = cv2.getTrackbarPos('grayscale','title')
    if gray == 1:
        frame = rgb2gray(frame)
        cv2.putText(frame, "grayscale", (20,160), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255),2)
        r = 0
        g = 0
        b = 0
    #edging
    sobel = cv2.getTrackbarPos('Sobel Filter:on/off','title')
    if sobel == 1:
        frame = rgb2gray(frame)
        frame = myfilter(Sx,Sy,frame)
        cv2.putText(frame, "sobel edging : ON", (20,180), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255),2)
    prewitt = cv2.getTrackbarPos('Prewitt Filter:on/off','title')
    if prewitt == 1:
        frame = rgb2gray(frame)
        frame = myfilter(Px,Py,frame)
        cv2.putText(frame, "prewitt edging : ON", (20,200), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255),2)
    #S Curve
    s = cv2.getTrackbarPos('s_curve','title')
    x = np.linspace(0, 255, 100)
    y = S_curve(x)
    realtime_graph(x,y)

    #S Curve on/off
    sw = cv2.getTrackbarPos('S Curve:on/off','title')
    if sw == 1:
        frame = S_curve(frame)
        cv2.putText(frame, "s curve contrast:ON (s:" + str(s)+")", (360,40), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0),2)
    cv2.putText(frame, "r:" + str(r)+"/100", (20,40), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,0,255),2)
    cv2.putText(frame, "g:" + str(g)+"/100", (20,80), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,255,0),2)
    cv2.putText(frame, "b:" + str(b)+"/100", (20,120), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,0,0),2)
    cv2.imshow('title', frame)  # show in the win

plt.close()
cap.release()
cv2.destroyAllWindows()

