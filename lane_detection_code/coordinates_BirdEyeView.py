# BirdEyeView 변환을 하기 위한
# 좌표를 찍어주는 코드

import cv2
import numpy as np
import pandas as pd

win_name = "scanning"
fname = "video_2"
img = cv2.imread("../image/" + fname + ".png")
rows, cols = img.shape[:2]
draw = img.copy()
pts_cnt = 0
pts = np.zeros((4, 2), dtype=np.float32)

def onMouse(event, x, y, flags, param):
    global pts_cnt
    # global camera_num
    if event == cv2.EVENT_LBUTTONDOWN:
        # 좌표에 초록색 동그라미 표시
        cv2.circle(draw, (x, y), 10, (0, 255, 0), -1)
        cv2.imshow(win_name, draw)

        # 마우스 좌표 저장
        pts[pts_cnt] = [x, y]
        pts_cnt += 1
        if pts_cnt == 4:
            # 좌표 4개 중 상하좌우 찾기
            sm = pts.sum(axis=1)  # 4쌍의 좌표 각각 x+y 계산
            diff = np.diff(pts, axis=1)  # 4쌍의 좌표 각각 x-y 계산

            topLeft = pts[np.argmin(sm)]  # x+y가 가장 값이 좌상단 좌표
            bottomRight = pts[np.argmax(sm)]  # x+y가 가장 큰 값이 우하단 좌표d
            topRight = pts[np.argmin(diff)]  # x-y가 가장 작은 것이 우상단 좌표
            bottomLeft = pts[np.argmax(diff)]  # x-y가 가장 큰 값이 좌하단 좌표

            print(topLeft)
            print(bottomRight)
            print(topRight)
            print(bottomLeft)

            vertices = np.array(
                [topLeft, bottomLeft, bottomRight, topRight], dtype=np.int32)

            print(vertices)
            # np.savetxt('../coordinates/'+camera_num+'.txt', vertices, fmt='%d', delimiter=' ')  # 좌표값 저장
            df = pd.DataFrame(vertices)
            df.to_csv('../coordinates/'+fname+'.csv', index=None)

            # 변환 전 4개 좌표
            pts1 = np.float32([topLeft, topRight, bottomRight, bottomLeft])

            # 변환 후 영상에 사용할 서류의 폭과 높이 계산
            w1 = abs(bottomRight[0] - bottomLeft[0])
            w2 = abs(topRight[0] - topLeft[0])
            h1 = abs(topRight[1] - bottomRight[1])
            h2 = abs(topLeft[1] - bottomLeft[1])
            width = max([w1, w2])  # 두 좌우 거리간의 최대값이 서류의 폭
            height = max([h1, h2])  # 두 상하 거리간의 최대값이 서류의 높이

            print(width)
            print(height)

            # 변환 후 4개 좌표
            pts2 = np.float32([[0, 0], [width - 1, 0],
                               [width - 1, height - 1], [0, height - 1]])

            # 변환 행렬 계산
            mtrx = cv2.getPerspectiveTransform(pts1, pts2)
            # 원근 변환 적용
            result = cv2.warpPerspective(img, mtrx, (int(width), int(height)))
            cv2.imshow('scanned', result)
            cv2.imwrite('../image/'+fname+'_BEV.jpg', result)


cv2.imshow(win_name, img)
cv2.setMouseCallback(win_name, onMouse)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 참고 사이트
# https://minimin2.tistory.com/135
# https://jdselectron.tistory.com/123
