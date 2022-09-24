# 차선 검출시 필터링을 통해
# 검출에 대한 정확도 향상을 위한 코드

import numpy as np
import cv2

video_file = '../video/test_2.mp4'

cap = cv2.VideoCapture(video_file) # 동영상 캡쳐 객체 생성  ---①

if cap.isOpened():                 # 캡쳐 객체 초기화 확인
    while True:
        ret, img = cap.read()      # 다음 프레임 읽기      --- ②

        if ret:                     # 프레임 읽기 정상

            ret, thr1 = cv2.threshold(img, 170, 255, cv2.THRESH_BINARY)
            ret, thr2 = cv2.threshold(img, 170, 255, cv2.THRESH_BINARY_INV)
            ret, thr3 = cv2.threshold(img, 170, 255, cv2.THRESH_TRUNC)
            ret, thr4 = cv2.threshold(img, 170, 255, cv2.THRESH_TOZERO)
            ret, thr5 = cv2.threshold(img, 170, 255, cv2.THRESH_TOZERO_INV)

            cv2.imshow('original', img)
            cv2.imshow('BINARY', thr1)
            cv2.imshow('BINARY_INV', thr2)
            # cv2.imshow('TRUNC', thr3)
            cv2.imshow('TOZERO', thr4)
            # cv2.imshow('TOZERO_INV', thr5)

            #cv2.imshow(video_file, img) # 화면에 표시  --- ③
            cv2.waitKey(25)            # 25ms 지연(40fps로 가정)   --- ④
        else:                       # 다음 프레임 읽을 수 없슴,
            break                   # 재생 완료
else:
    print("can't open video.")      # 캡쳐 객체 초기화 실패
cap.release()                       # 캡쳐 자원 반납
cv2.destroyAllWindows()

