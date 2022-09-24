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


# def hough_to(img, lines):
#     for line in lines:
#         for rho, theta in line:
#             a = np.cos(theta)
#             b = np.sin(theta)
#             x0 = a * rho
#             y0 = b * rho
#             x1 = int(x0 + 1000 * (-b))
#             y1 = int(y0 + 1000 * (a))
#             x2 = int(x0 - 1000 * (-b))
#             y2 = int(y0 - 1000 * (a))
#
#             # hough_transe.extend([][][x1,y1,x2,y2])
#             cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)


def draw_fit_line(img, lines, color=[255, 0, 0], thickness=10):  # 대표선 그리기
    cv2.line(img, (lines[0], lines[1]), (lines[2], lines[3]), color, thickness)


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

    # line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # draw_lines(line_img, lines)

    # print("lines1: ", lines)
    #print("lines.shape", lines.shape)
    #print("lines.ndim", lines.ndim)


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

    #         lines[line] = [[x1,y1][x2,y2]]

    # print("lines2: ", houghLine)
    # [[[x1,y1]

    return houghLine
    #return lines


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

print("slope", slope_degree)
print("slope", slope_degree>-20)
# 필터링된 직선 버리기
# L_lines, R_lines = line_arr[(slope_degree>0),:], line_arr[(slope_degree<0),:]
lines = line_arr[(slope_degree>-20),:]
# 현재는 수직에 가까운 중앙선 검출을 위한 값을 입력 해놓은 상태


temp = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
lines = lines[:,None]

# 직선 그리기
draw_lines(temp, lines)
result = weighted_img(temp, image)  # 원본 이미지에 검출된 선 overlap

cv2.imshow('canny', canny_img)
cv2.imshow('result', result)  # 결과 이미지 출력
# result = weighted_img(line_arr, image)
# cv2.imshow('result', result)
cv2.waitKey(0)
