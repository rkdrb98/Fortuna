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

# #재생할 파일의 넓이와 높이
# width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

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

    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')  #mp4 형식의 영상을 저장하기 위해서
    out = cv2.VideoWriter('../video/BEV_' + fname + '-1.mp4', fourcc, fps, (int(width), int(height)))  # isColor=True

    while True:
        ret, img = cap.read()  # 다음 프레임 읽기      --- ②
        # ret - 읽기 성공시 True, 실패시 False
        # img - 읽어온 이미지

        if ret:  # 프레임 읽기 정상
            # 원근 변환 적용
            result = cv2.warpPerspective(img, mtrx, (int(width), int(height)))

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
