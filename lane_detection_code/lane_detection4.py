# -*- coding: cp949 -*-
# -*- coding: utf-8 -*- # �ѱ� �ּ������� �̰� �ؾ���

# Bird Eye View�� ��ȯ�� �̹����� ���� �����ϴ� �ڵ�

import cv2  # opencv ���
import numpy as np
from pandas import read_csv


def grayscale(img):  # ����̹����� ��ȯ
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def canny(img, low_threshold, high_threshold):  # Canny �˰���
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):  # ����þ� ����
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def draw_lines(img, lines, color=[0, 0, 255], thickness=10):  # �� �׸���
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold):  # ���� ��ȯ , min_line_len, max_line_gap
    houghLine= np.empty((0,4),int)
    # lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
    #                         maxLineGap=max_line_gap)
    lines = cv2.HoughLines(img, rho, theta, threshold)
    # cv2.HoughLinesP(image, rho, theta, threshold, lines=None, minLineLength=None, maxLineGap=None)
    # ? image: �Է� ���� ����
    # ? rho: ���� �迭���� rho ���� ����. (e.g.) 1.0 �� 1�ȼ� ����
    # ? theta: ���� �迭���� theta ���� ����. (e.g.) np.pi / 180 �� 1? ����.
    # ? threshold: ���� �迭���� �������� �Ǵ��� �Ӱ谪, ������ ũ��� �� ������� �ݺ��, ��Ȯ���ʹ� ���
    # ? lines: ������ ���۰� �� ��ǥ(x1, y1, x2, y2) ������ ��� �ִ� numpy.ndarray. shape=(N, 1, 4). dtype=numpy.int32.
    # ? minLineLength: ������ ������ �ּ� ����
    # ? maxLineGap: �������� ������ �ִ� ���� �� ����

    # HoughLines - ���� ���, ������������ ���� �׸� �� ����, ���� ���Ѵ�
    # HoughLinesP - ���� ���, ��ǥ���� ���, ���� �ּ� ���̳� �� ������ �ִ� ���� ���� �ʿ�

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

            # print("X, Y��: ", x1,y1,x2,y2)

            coo = np.array([[x1,y1,x2,y2]])
            houghLine = np.append(houghLine, coo, axis=0)

    return houghLine


def weighted_img(img, initial_img, ��=1, ��=1., ��=0.):  # �� �̹��� operlap �ϱ�
    return cv2.addWeighted(initial_img, ��, img, ��, ��)


image = cv2.imread('../image/video_2_BEV.jpg')  # �̹��� �б�
height, width = image.shape[:2]  # �̹��� ����, �ʺ�

gray_img = grayscale(image)  # ����̹����� ��ȯ
blur_img = gaussian_blur(gray_img, 3)  # Blur ȿ��

canny_img = canny(blur_img, 90, 110)  # Canny edge �˰���

# line_arr = hough_lines(canny_img, 1, 1 * np.pi / 180, 30, 11, 20)  # ���� ��ȯ
line_arr = hough_lines(canny_img, 1, 1 * np.pi / 180, 42)  # ���� ��ȯ
# print("�迭Ȯ��: ", line_arr.shape)
# print("1: ", line_arr)
# line_arr = np.squeeze(line_arr)  # ũ�Ⱑ 1�� axis ����
#
# ���� ���ϱ�
slope_degree = (np.arctan2(line_arr[:, 1] - line_arr[:, 3], line_arr[:, 0] - line_arr[:, 2]) * 180) / np.pi
# print(slope_degree)

# ���� ���� ����
# print('���밪', np.abs(slope_degree) < 100)
horizontal = 100  # default: 160
line_arr = line_arr[np.abs(slope_degree) < horizontal]  # np.abs: ���밪 ���ϴ� �Լ�
slope_degree = slope_degree[np.abs(slope_degree) < horizontal]

# ���� ���� ����
verticality = 85  # default: 95
line_arr = line_arr[np.abs(slope_degree) > verticality]
slope_degree = slope_degree[np.abs(slope_degree) > verticality]

# ���͸��� ���� ������
average = sum(slope_degree)
line_average = average/len(slope_degree)
print(line_average)

if line_average <= 0:
    arr = line_arr[(slope_degree>line_average),:]
    print(">")
else:
    arr = line_arr[(slope_degree<line_average),:]
    print("<")

arr = arr[arr[:, 0].argsort()]  # �ߺ��Ǵ� ������ �������ֱ� ���� x1�� �������� ����

# ���� ������ 50�� ���� ������ ����
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

# ���� �׸���
draw_lines(temp, lines)
result = weighted_img(temp, image)  # ���� �̹����� ����� �� overlap

cv2.imshow('canny', canny_img)
cv2.imshow('result', result)  # ��� �̹��� ���
cv2.waitKey(0)
