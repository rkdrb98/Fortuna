# -*- coding: cp949 -*-
# -*- coding: utf-8 -*- # �ѱ� �ּ������� �̰� �ؾ���

# �߾Ӽ�(�����) �����ϴ� �ڵ�
# �Լ� HoughLinesP()�� ���

import cv2  # opencv ���
import numpy as np
from pandas import read_csv


def hsv_fillter(img):  # HSV ���� ����
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


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


def draw_fit_line(img, lines, color=[255, 0, 0], thickness=10):  # ��ǥ�� �׸���
    cv2.line(img, (lines[0], lines[1]), (lines[2], lines[3]), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):  # ���� ��ȯ
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    # cv2.HoughLinesP(image, rho, theta, threshold, lines=None, minLineLength=None, maxLineGap=None)
    # ? image: �Է� ���� ����
    # ? rho: ���� �迭���� rho ���� ����. (e.g.) 1.0 �� 1�ȼ� ����
    # ? theta: ���� �迭���� theta ���� ����. (e.g.) np.pi / 180 �� 1? ����.
    # ? threshold: ���� �迭���� �������� �Ǵ��� �Ӱ谪
    # ? lines: ������ ���۰� �� ��ǥ(x1, y1, x2, y2) ������ ��� �ִ� numpy.ndarray. shape=(N, 1, 4). dtype=numpy.int32.
    # ? minLineLength: ������ ������ �ּ� ����
    # ? maxLineGap: �������� ������ �ִ� ���� �� ����

    # HoughLines - ���� ���, ������������ ���� �׸� �� ����, ���� ���Ѵ�
    # HoughLinesP - ���� ���, ��ǥ���� ���, ���� �ּ� ���̳� �� ������ �ִ� ���� ���� �ʿ�

    # print("lines: ", lines)

    # line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # draw_lines(line_img, lines)

    print("lines1: ", lines)
    print("lines.shape", lines.shape)
    print("lines.ndim", lines.ndim)

    return lines


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


image = cv2.imread('../image/video_1.jpg')  # �̹��� �б�
height, width = image.shape[:2]  # �̹��� ����, �ʺ�

gray_img = grayscale(image)  # ����̹����� ��ȯ

hsvLower = np.array([30-30, 20, 20])    # ������ ���� ����(HSV)
hsvUpper = np.array([30, 255, 255])    # ������ ���� ����(H(����)S(ä��)V(��))

hsv_img = hsv_fillter(image)
hsv_mask = cv2.inRange(hsv_img, hsvLower, hsvUpper)    # HSV���� ����ũ�� �ۼ�
hsv_result = cv2.bitwise_and(image, image, mask=hsv_mask) # ���� �̹����� ����ũ�� �ռ�

blur_img = gaussian_blur(hsv_result, 3)  # Blur ȿ��
# blur_img = gaussian_blur(gray_img, 3)  # Blur ȿ��

canny_img = canny(blur_img, 5, 180)  # Canny edge �˰���

# vertices = np.array(
#     [[(50, height), (width / 2 - 45, height / 2 + 60), (width / 2 + 45, height / 2 + 60), (width - 50, height)]],
#     dtype=np.int32)

# ROI���� �� ���θ� ���� �ʿ�� ������ �ִ��� �߾Ӽ��� ǥ���� �� �ִ� ������ �簢������ �����غ���

df = read_csv('../coordinates/test2.csv')
vertices = [df.values]
ROI_img = region_of_interest(canny_img, vertices)  # ROI ����

line_arr = hough_lines(ROI_img, 1, 1 * np.pi / 180, 30, 10, 20)  # ���� ��ȯ
# line_arr = hough_lines(canny_img, 1, 1 * np.pi / 180, 30, 10, 20)  # ���� ��ȯ
# print("1: ", line_arr)
line_arr = np.squeeze(line_arr)  # ũ�Ⱑ 1�� axis ����
print("�迭Ȯ��: ", line_arr.shape)
print("2: ", line_arr)

# ���� ���ϱ�
slope_degree = (np.arctan2(line_arr[:, 1] - line_arr[:, 3], line_arr[:, 0] - line_arr[:, 2]) * 180) / np.pi
# print(slope_degree)

# ���� ���� ����
line_arr = line_arr[np.abs(slope_degree) < 130]  # default: 160
slope_degree = slope_degree[np.abs(slope_degree) < 130]
# ���� ���� ����
line_arr = line_arr[np.abs(slope_degree) > 88]  # default: 95
slope_degree = slope_degree[np.abs(slope_degree) > 88]
# ���͸��� ���� ������
# L_lines, R_lines = line_arr[(slope_degree>0),:], line_arr[(slope_degree<0),:]
lines = line_arr[(slope_degree<0),:]
# �̰Ÿ� if ������ �̿��ؼ� �߾Ӽ��� ���⿡ ���� ROI ���� ������ ���ָ� �ɵ�
# ����� ������ ����� �߾Ӽ� ������ ���� ���� �Է� �س��� ����

temp = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
lines = lines[:,None]
# # ���� �׸���
# draw_lines(temp, L_lines)
# draw_lines(temp, lines)

# �߾Ӽ� ��ǥ�� ���ϱ�
fit_line = get_fitline(image, lines)

# ��ǥ�� �׸���
draw_fit_line(temp, fit_line)

result = weighted_img(temp, image)  # ���� �̹����� ����� �� overlap

# cv2.imshow('canny', canny_img)
cv2.imshow('result', result)  # ��� �̹��� ���
cv2.waitKey(0)

# ���� ����Ʈ
# https://blog.naver.com/windowsub0406/220894645729
