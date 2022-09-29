# 선이 그어진 버드아이뷰 영상을 저장

import cv2
import numpy as np
from pandas import read_csv


win_name = "bird eye view"
fname = "video_2"
filename = "../video/" + fname + ".mp4"
cap = cv2.VideoCapture(filename)

fps = cap.get(cv2.CAP_PROP_FPS)
print('fps: ', fps)


def grayscale(img):  # 흑백이미지로 변환
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def canny(img, low_threshold, high_threshold):  # Canny 알고리즘
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):  # 가우시안 필터
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

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


def draw_lines(img, lines, color=[0, 0, 255], thickness=10):  # 선 그리기
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def weighted_img(img, initial_img, α=1, β=1., λ=0.):  # 두 이미지 operlap 하기
    return cv2.addWeighted(initial_img, α, img, β, λ)

def lane_detection(img):
    image = img  # 이미지 읽기

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
    horizontal = 95  # default: 160
    line_arr = line_arr[np.abs(slope_degree) < horizontal]  # np.abs: 절대값 구하는 함수
    slope_degree = slope_degree[np.abs(slope_degree) < horizontal]

    # 수직 기울기 제한
    verticality = 85  # default: 95
    line_arr = line_arr[np.abs(slope_degree) > verticality]
    slope_degree = slope_degree[np.abs(slope_degree) > verticality]

    print("slope", slope_degree)
    # print("slope", slope_degree>-20)

    # 필터링된 직선 버리기
    # L_lines, R_lines = line_arr[(slope_degree>0),:], line_arr[(slope_degree<0),:]
    average = sum(slope_degree)
    line_average = average / len(slope_degree)
    print(line_average)

    if line_average <= 0:
        arr = line_arr[(slope_degree < 0), :]  # 왼쪽 기울기(-)
        # print(">")
    else:
        arr = line_arr[(slope_degree > 0), :]  # 오른쪽 기울기(+)
    # print("<")
    # 현재는 수직에 가까운 중앙선 검출을 위한 값을 입력 해놓은 상태
    # print("1: ", arr)
    arr = arr[arr[:, 0].argsort()]  # 중복되는 선들을 제거해주기 위해 x1을 기준으로 정렬

    # 선의 간격이 50이 넘지 않으면 삭제
    x1 = arr[0][0]
    lines = arr[:]
    num = 1
    for line in arr[1:, 0]:
        if x1 + 100 < line:
            x1 = line
            num += 1
            # print(line)
        else:
            lines = np.delete(lines, num, axis=0)
            # print("2: ", lines)

    # print("3: ", lines)
    temp = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    lines = lines[:, None]

    # # 중앙선 대표선 구하기
    # fit_line = get_fitline(image, lines)
    #
    # # 대표선 그리기
    # draw_fit_line(temp, fit_line)

    # 직선 그리기
    draw_lines(temp, lines)
    # result = weighted_img(temp, image)  # 원본 이미지에 검출된 선 overlap

    return lines, temp


def BirdEyeView():
    df = read_csv('../coordinates/' + fname + '.csv')
    vertices = [df.values]

    topLeft = vertices[0][0]
    bottomLeft = vertices[0][1]
    bottomRight = vertices[0][2]
    topRight = vertices[0][3]

    # print(topLeft)
    # print(bottomLeft)
    # print(bottomRight)
    # print(topRight)


    # 변환 전 4개 좌표
    pts1 = np.float32([topLeft, topRight, bottomRight, bottomLeft])

    # 변환 후 영상에 사용할 서류의 폭과 높이 계산
    w1 = abs(bottomRight[0] - bottomLeft[0])
    w2 = abs(topRight[0] - topLeft[0])
    h1 = abs(topRight[1] - bottomRight[1])
    h2 = abs(topLeft[1] - bottomLeft[1])
    width = max([w1, w2])  # 두 좌우 거리간의 최대값이 서류의 폭
    height = max([h1, h2])  # 두 상하 거리간의 최대값이 서류의 높이

    # print(width)
    # print(height)

    # 변환 후 4개 좌표
    pts2 = np.float32([[0, 0], [width - 1, 0],
                       [width - 1, height - 1], [0, height - 1]])

    # 변환 행렬 계산
    mtrx = cv2.getPerspectiveTransform(pts1, pts2)

    return mtrx, width, height




if cap.isOpened():
    mtrx, width, height = BirdEyeView()
    ret, img = cap.read()  # 다음 프레임 읽기      --- ②
    # ret - 읽기 성공시 True, 실패시 False
    # img - 읽어온 이미지

    if ret:  # 프레임 읽기 정상
        # 원근 변환 적용
        image = cv2.warpPerspective(img, mtrx, (int(width), int(height)))
        lines, temp = lane_detection(image)
        # temp = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        draw_lines(temp, lines)


    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    out = cv2.VideoWriter('../video/BEV_' + fname + '-3.mp4', fourcc, fps, (int(width), int(height)))  # isColor=True

    while True:
        ret, img = cap.read()  # 다음 프레임 읽기      --- ②
        # ret - 읽기 성공시 True, 실패시 False
        # img - 읽어온 이미지

        if ret:  # 프레임 읽기 정상
            # 원근 변환 적용
            image = cv2.warpPerspective(img, mtrx, (int(width), int(height)))
            # result, temp = lane_detection(image)
            result = weighted_img(temp, image)

            cv2.imshow('scanned', result)
            out.write(result)
            # cv2.imshow(filename, img)  # 화면에 표시  --- ③
            cv2.waitKey(25)  # 25ms 지연(40fps로 가정)   --- ④


        else:  # 다음 프레임 읽을 수 없슴,
            print("end")
            break  # 재생 완료
    else:
        print("can't open video.")

cap.release()
out.release()
cv2.waitKey(0)
cv2.destroyAllWindows()
