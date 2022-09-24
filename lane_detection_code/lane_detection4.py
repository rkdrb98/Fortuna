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


def region_of_interest(img, vertices, color3=(255, 255, 255), color1=255):  # ROI ����

    mask = np.zeros_like(img)  # mask = img�� ���� ũ���� �� �̹���

    if len(img.shape) > 2:  # Color �̹���(3ä��)��� :
        color = color3
    else:  # ��� �̹���(1ä��)��� :
        color = color1

    # vertices�� ���� ����� �̷��� �ٰ����κ�(ROI �����κ�)�� color�� ä��
    cv2.fillPoly(mask, vertices, color)

    # �̹����� color�� ä���� ROI�� ��ħ
    ROI_image = cv2.bitwise_and(img, mask)
    return ROI_image


def draw_lines(img, lines, color=[0, 0, 255], thickness=2):  # �� �׸���
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


def draw_fit_line(img, lines, color=[255, 0, 0], thickness=10):  # ��ǥ�� �׸���
    cv2.line(img, (lines[0], lines[1]), (lines[2], lines[3]), color, thickness)


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

            # print("X, Y��: ", x1,y1,x2,y2)

            coo = np.array([[x1,y1,x2,y2]])
            houghLine = np.append(houghLine, coo, axis=0)

    #         lines[line] = [[x1,y1][x2,y2]]

    # print("lines2: ", houghLine)
    # [[[x1,y1]

    return houghLine
    #return lines


def weighted_img(img, initial_img, ��=1, ��=1., ��=0.):  # �� �̹��� operlap �ϱ�
    return cv2.addWeighted(initial_img, ��, img, ��, ��)


def get_fitline(img, f_lines):  # ��ǥ�� ���ϱ�
    lines = np.squeeze(f_lines)
    lines = lines.reshape(lines.shape[0] * 2, 2)
    rows, cols = img.shape[:2]
    # cv2.fitLine - �־��� ���� ������ ������ ��ȯ
    # fitLine(points, distType, param, reps, aeps, line)
    # distType: �Ÿ� ��� ��� (cv2.DIST_L2, cv2.DIST_L1, cv2.DIST_L12, cv2.DIST_FAIR, cv2.DIST_WELSCH, cv2.DIST_HUBER)
    # param: distType�� ������ ����, 0 = ���� �� ����
    # reps: ������ ��Ȯ��, ���� ���� ��ǥ�� �Ÿ�, 0.01 ����
    # aeps: ���� ��Ȯ��, 0.01 ����
    # line(optional): vx, vy ����ȭ�� ���� ����, x0, y0: �߽��� ��ǥ

    output = cv2.fitLine(lines, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x, y = output[0], output[1], output[2], output[3]
    x1, y1 = int(((img.shape[0] - 1) - y) / vy * vx + x), img.shape[0] - 1
    x2, y2 = int(((img.shape[0] / 2 + 100) - y) / vy * vx + x), int(img.shape[0] / 2 + 100)

    result = [x1, y1, x2, y2]
    return result


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

print("slope", slope_degree)
print("slope", slope_degree>-20)
# ���͸��� ���� ������
# L_lines, R_lines = line_arr[(slope_degree>0),:], line_arr[(slope_degree<0),:]
lines = line_arr[(slope_degree>-20),:]
# ����� ������ ����� �߾Ӽ� ������ ���� ���� �Է� �س��� ����


temp = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
lines = lines[:,None]

# ���� �׸���
draw_lines(temp, lines)
result = weighted_img(temp, image)  # ���� �̹����� ����� �� overlap

cv2.imshow('canny', canny_img)
cv2.imshow('result', result)  # ��� �̹��� ���
# result = weighted_img(line_arr, image)
# cv2.imshow('result', result)
cv2.waitKey(0)
