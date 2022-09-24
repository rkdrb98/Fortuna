# -*- coding: cp949 -*-
# -*- coding: utf-8 -*- # 한글 주석쓰려면 이거 해야함

# 중앙선(노란선) 검출하는 코드
# 함수 HoughLinesP()를 사용

import cv2  # opencv 사용
import numpy as np
from pandas import read_csv


def hsv_fillter(img):  # HSV 필터 적용
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


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


def draw_fit_line(img, lines, color=[255, 0, 0], thickness=10):  # 대표선 그리기
    cv2.line(img, (lines[0], lines[1]), (lines[2], lines[3]), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):  # 허프 변환
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    # cv2.HoughLinesP(image, rho, theta, threshold, lines=None, minLineLength=None, maxLineGap=None)
    # ? image: 입력 에지 영상
    # ? rho: 축적 배열에서 rho 값의 간격. (e.g.) 1.0 → 1픽셀 간격
    # ? theta: 축적 배열에서 theta 값의 간격. (e.g.) np.pi / 180 → 1? 간격.
    # ? threshold: 축적 배열에서 직선으로 판단할 임계값
    # ? lines: 선분의 시작과 끝 좌표(x1, y1, x2, y2) 정보를 담고 있는 numpy.ndarray. shape=(N, 1, 4). dtype=numpy.int32.
    # ? minLineLength: 검출할 선분의 최소 길이
    # ? maxLineGap: 직선으로 간주할 최대 에지 점 간격

    # HoughLines - 직선 출력, 직선방정식을 구해 그릴 수 있음, 길이 무한대
    # HoughLinesP - 선분 출력, 좌표값을 출력, 선의 최소 길이나 점 사이의 최대 간격 설정 필요

    # print("lines: ", lines)

    # line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # draw_lines(line_img, lines)

    print("lines1: ", lines)
    print("lines.shape", lines.shape)
    print("lines.ndim", lines.ndim)

    return lines


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

hsvLower = np.array([30-30, 20, 20])    # 추출할 색의 하한(HSV)
hsvUpper = np.array([30, 255, 255])    # 추출할 색의 상한(H(색조)S(채도)V(명도))

hsv_img = hsv_fillter(image)
hsv_mask = cv2.inRange(hsv_img, hsvLower, hsvUpper)    # HSV에서 마스크를 작성
hsv_result = cv2.bitwise_and(image, image, mask=hsv_mask) # 원래 이미지와 마스크를 합성

blur_img = gaussian_blur(hsv_result, 3)  # Blur 효과
# blur_img = gaussian_blur(gray_img, 3)  # Blur 효과

canny_img = canny(blur_img, 5, 180)  # Canny edge 알고리즘

# vertices = np.array(
#     [[(50, height), (width / 2 - 45, height / 2 + 60), (width / 2 + 45, height / 2 + 60), (width - 50, height)]],
#     dtype=np.int32)

# ROI값은 딱 도로만 잡을 필요는 없으니 최대한 중앙선만 표시할 수 있는 정도의 사각형으로 지정해보기

df = read_csv('../coordinates/test2.csv')
vertices = [df.values]
ROI_img = region_of_interest(canny_img, vertices)  # ROI 설정

line_arr = hough_lines(ROI_img, 1, 1 * np.pi / 180, 30, 10, 20)  # 허프 변환
# line_arr = hough_lines(canny_img, 1, 1 * np.pi / 180, 30, 10, 20)  # 허프 변환
# print("1: ", line_arr)
line_arr = np.squeeze(line_arr)  # 크기가 1인 axis 제거
print("배열확인: ", line_arr.shape)
print("2: ", line_arr)

# 기울기 구하기
slope_degree = (np.arctan2(line_arr[:, 1] - line_arr[:, 3], line_arr[:, 0] - line_arr[:, 2]) * 180) / np.pi
# print(slope_degree)

# 수평 기울기 제한
line_arr = line_arr[np.abs(slope_degree) < 130]  # default: 160
slope_degree = slope_degree[np.abs(slope_degree) < 130]
# 수직 기울기 제한
line_arr = line_arr[np.abs(slope_degree) > 88]  # default: 95
slope_degree = slope_degree[np.abs(slope_degree) > 88]
# 필터링된 직선 버리기
# L_lines, R_lines = line_arr[(slope_degree>0),:], line_arr[(slope_degree<0),:]
lines = line_arr[(slope_degree<0),:]
# 이거를 if 문으로 이용해서 중앙선의 기울기에 따른 ROI 영역 지정을 해주면 될듯
# 현재는 수직에 가까운 중앙선 검출을 위한 값을 입력 해놓은 상태

temp = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
lines = lines[:,None]
# # 직선 그리기
# draw_lines(temp, L_lines)
# draw_lines(temp, lines)

# 중앙선 대표선 구하기
fit_line = get_fitline(image, lines)

# 대표선 그리기
draw_fit_line(temp, fit_line)

result = weighted_img(temp, image)  # 원본 이미지에 검출된 선 overlap

# cv2.imshow('canny', canny_img)
cv2.imshow('result', result)  # 결과 이미지 출력
cv2.waitKey(0)

# 참고 사이트
# https://blog.naver.com/windowsub0406/220894645729
