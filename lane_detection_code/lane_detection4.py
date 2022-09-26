# -*- coding: cp949 -*-
# -*- coding: utf-8 -*- # 한글 주석쓰려면 이거 해야함

# Bird Eye View로 변환한 이미지로 차선 검출하는 코드

import cv2  # opencv 사용
import numpy as np
from pandas import read_csv


def grayscale(img):  # 흑백이미지로 변환
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def canny(img, low_threshold, high_threshold):  # Canny 알고리즘
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):  # 가우시안 필터
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def draw_lines(img, lines, color=[0, 0, 255], thickness=10):  # 선 그리기
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold):  # 허프 변환 , min_line_len, max_line_gap
    houghLine= np.empty((0,4),int)
    # lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
    #                         maxLineGap=max_line_gap)
    lines = cv2.HoughLines(img, rho, theta, threshold)
    # cv2.HoughLinesP(image, rho, theta, threshold, lines=None, minLineLength=None, maxLineGap=None)
    # ? image: 입력 에지 영상
    # ? rho: 축적 배열에서 rho 값의 간격. (e.g.) 1.0 → 1픽셀 간격
    # ? theta: 축적 배열에서 theta 값의 간격. (e.g.) np.pi / 180 → 1? 간격.
    # ? threshold: 축적 배열에서 직선으로 판단할 임계값, 숫자의 크기와 선 검출수는 반비례, 정확도와는 비례
    # ? lines: 선분의 시작과 끝 좌표(x1, y1, x2, y2) 정보를 담고 있는 numpy.ndarray. shape=(N, 1, 4). dtype=numpy.int32.
    # ? minLineLength: 검출할 선분의 최소 길이
    # ? maxLineGap: 직선으로 간주할 최대 에지 점 간격

    # HoughLines - 직선 출력, 직선방정식을 구해 그릴 수 있음, 길이 무한대
    # HoughLinesP - 선분 출력, 좌표값을 출력, 선의 최소 길이나 점 사이의 최대 간격 설정 필요

    for line in lines:
        for rho, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            # print("X, Y값: ", x1,y1,x2,y2)

            coo = np.array([[x1,y1,x2,y2]])
            houghLine = np.append(houghLine, coo, axis=0)

    return houghLine


def weighted_img(img, initial_img, α=1, β=1., λ=0.):  # 두 이미지 operlap 하기
    return cv2.addWeighted(initial_img, α, img, β, λ)


image = cv2.imread('../image/video_2_BEV.jpg')  # 이미지 읽기
height, width = image.shape[:2]  # 이미지 높이, 너비

gray_img = grayscale(image)  # 흑백이미지로 변환
blur_img = gaussian_blur(gray_img, 3)  # Blur 효과

canny_img = canny(blur_img, 90, 110)  # Canny edge 알고리즘

# line_arr = hough_lines(canny_img, 1, 1 * np.pi / 180, 30, 11, 20)  # 허프 변환
line_arr = hough_lines(canny_img, 1, 1 * np.pi / 180, 42)  # 허프 변환
# print("배열확인: ", line_arr.shape)
# print("1: ", line_arr)
# line_arr = np.squeeze(line_arr)  # 크기가 1인 axis 제거
#
# 기울기 구하기
slope_degree = (np.arctan2(line_arr[:, 1] - line_arr[:, 3], line_arr[:, 0] - line_arr[:, 2]) * 180) / np.pi
# print(slope_degree)

# 수평 기울기 제한
# print('절대값', np.abs(slope_degree) < 100)
horizontal = 100  # default: 160
line_arr = line_arr[np.abs(slope_degree) < horizontal]  # np.abs: 절대값 구하는 함수
slope_degree = slope_degree[np.abs(slope_degree) < horizontal]

# 수직 기울기 제한
verticality = 85  # default: 95
line_arr = line_arr[np.abs(slope_degree) > verticality]
slope_degree = slope_degree[np.abs(slope_degree) > verticality]

# 필터링된 직선 버리기
average = sum(slope_degree)
line_average = average/len(slope_degree)
print(line_average)

if line_average <= 0:
    arr = line_arr[(slope_degree>line_average),:]
    print(">")
else:
    arr = line_arr[(slope_degree<line_average),:]
    print("<")

arr = arr[arr[:, 0].argsort()]  # 중복되는 선들을 제거해주기 위해 x1을 기준으로 정렬

# 선의 간격이 50이 넘지 않으면 삭제
x1 = arr[0][0]
lines = arr[:]
num = 1
for line in arr[1:,0]:
    if x1+50 < line:
        x1 = line
        num += 1
        # print(line)
    else:
        lines = np.delete(lines,num,axis=0)
        # print("2: ", lines)

temp = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
lines = lines[:,None]

# 직선 그리기
draw_lines(temp, lines)
result = weighted_img(temp, image)  # 원본 이미지에 검출된 선 overlap

cv2.imshow('canny', canny_img)
cv2.imshow('result', result)  # 결과 이미지 출력
cv2.waitKey(0)
