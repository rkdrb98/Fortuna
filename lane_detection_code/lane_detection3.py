# -*- coding: cp949 -*-
# -*- coding: utf-8 -*- # 한글 주석쓰려면 이거 해야함

# Bird Eye View로 변환한 이미지로 차선 검출하는 코드
# 기울기에 대한 제한은 아직 안한 상태
# 함수 HoughLines()를 사용

import cv2  # opencv 사용
import numpy as np
from pandas import read_csv


def grayscale(img):  # 흑백이미지로 변환
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def canny(img, low_threshold, high_threshold):  # Canny 알고리즘
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):  # 가우시안 필터
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices, color3=(255, 255, 255), color1=255):  # ROI 셋팅

    mask = np.zeros_like(img)  # mask = img와 같은 크기의 빈 이미지

    if len(img.shape) > 2:  # Color 이미지(3채널)라면 :
        color = color3
    else:  # 흑백 이미지(1채널)라면 :
        color = color1

    # vertices에 정한 점들로 이뤄진 다각형부분(ROI 설정부분)을 color로 채움
    cv2.fillPoly(mask, vertices, color)

    # 이미지와 color로 채워진 ROI를 합침
    ROI_image = cv2.bitwise_and(img, mask)
    return ROI_image


def draw_lines(img, lines, color=[0, 0, 255], thickness=2):  # 선 그리기
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_to(img, lines):
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

            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)


def draw_fit_line(img, lines, color=[255, 0, 0], thickness=10):  # 대표선 그리기
    cv2.line(img, (lines[0], lines[1]), (lines[2], lines[3]), color, thickness)


def hough_lines(img, rho, theta, threshold):  # 허프 변환 , min_line_len, max_line_gap
    # lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
    #                         maxLineGap=max_line_gap)
    lines = cv2.HoughLines(img, rho, theta, threshold)
    # cv2.HoughLinesP(image, rho, theta, threshold, lines=None, minLineLength=None, maxLineGap=None)
    # ? image: 입력 에지 영상
    # ? rho: 축적 배열에서 rho 값의 간격. (e.g.) 1.0 → 1픽셀 간격
    # ? theta: 축적 배열에서 theta 값의 간격. (e.g.) np.pi / 180 → 1? 간격.


    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # draw_lines(line_img, lines)

    hough_to(line_img, lines)

    # print("lines: ", lines)

    return line_img


def weighted_img(img, initial_img, α=1, β=1., λ=0.):  # 두 이미지 operlap 하기
    return cv2.addWeighted(initial_img, α, img, β, λ)


def get_fitline(img, f_lines):  # 대표선 구하기
    lines = np.squeeze(f_lines)
    lines = lines.reshape(lines.shape[0] * 2, 2)
    rows, cols = img.shape[:2]
    # cv2.fitLine - 주어진 점에 적합하 직선을 반환
    # fitLine(points, distType, param, reps, aeps, line)
    # distType: 거리 계산 방식 (cv2.DIST_L2, cv2.DIST_L1, cv2.DIST_L12, cv2.DIST_FAIR, cv2.DIST_WELSCH, cv2.DIST_HUBER)
    # param: distType에 전달할 인자, 0 = 최적 값 선택
    # reps: 반지름 정확도, 선과 원본 좌표의 거리, 0.01 권장
    # aeps: 각도 정확도, 0.01 권장
    # line(optional): vx, vy 정규화된 단위 벡터, x0, y0: 중심점 좌표

    output = cv2.fitLine(lines, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x, y = output[0], output[1], output[2], output[3]
    x1, y1 = int(((img.shape[0] - 1) - y) / vy * vx + x), img.shape[0] - 1
    x2, y2 = int(((img.shape[0] / 2 + 100) - y) / vy * vx + x), int(img.shape[0] / 2 + 100)

    result = [x1, y1, x2, y2]
    return result


image = cv2.imread('../image/video_1.jpg')  # 이미지 읽기
height, width = image.shape[:2]  # 이미지 높이, 너비

gray_img = grayscale(image)  # 흑백이미지로 변환
blur_img = gaussian_blur(gray_img, 3)  # Blur 효과

canny_img = canny(blur_img, 80, 115)  # Canny edge 알고리즘

# line_arr = hough_lines(canny_img, 1, 1 * np.pi / 180, 30, 11, 20)  # 허프 변환
line_arr = hough_lines(canny_img, 1, 1 * np.pi / 180, 30)  # 허프 변환
# print("1: ", line_arr)
# line_arr = np.squeeze(line_arr)  # 크기가 1인 axis 제거
# print("2: ", line_arr)

result = weighted_img(line_arr, image)
cv2.imshow('result', result)
cv2.waitKey(0)
