from bs4 import BeautifulSoup
import urllib.request
import lxml
import requests
import time


def get_eminem_video_link(target_url):
    
    response = requests.get(target_url)
    soup = BeautifulSoup(response.text, "lxml")
    lis = soup.find_all('div', {'class' : 'mt5 cctvView'}) # 크롤링 하고자하는 대상의 태그 위치 ( div 아래 mt5 cctvView ) 

    for li in lis :
        cctv_name = li.find('video', {'src' : True})['src'] # 해당 클래스의 scr값 저장
        print(cctv_name)
        
        savename =  cctv_name.split('/')[5].split('.')[1]
        print(savename)
        urllib.request.urlretrieve(cctv_name, savename+'.mp4') #비디오 저장
        print("저장완료")
    


    


while(1):

    target_url = "http://traffic.daejeon.go.kr/map/trafficInfo/cctvCk.do?cctvId=CCTV02" #cctv영상 링크
    get_eminem_video_link(target_url)
    time.sleep(150) # 대전 교통정보시스템 기준 2분30초마다 업데이트 , 타임슬립 150초 지
