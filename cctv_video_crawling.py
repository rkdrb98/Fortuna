from bs4 import BeautifulSoup
import urllib.request
import lxml
import requests
import time
import os

def get_eminem_video_link(target_url):
    
    response = requests.get(target_url)
    soup = BeautifulSoup(response.text, "lxml")
    lis = soup.find_all('div', {'class' : 'mt5 cctvView'}) # 크롤링 하고자하는 대상의 태그 위치 ( div 아래 mt5 cctvView ) 

    for li in lis :
        cctv_name = li.find('video', {'src' : True})['src'] # 해당 클래스의 scr값 저장
        print(cctv_name)
        
        savename =  cctv_name.split('/')[5] #저장되는 이름
        print(savename)
        
        path = "CCTV_Video"+"/"+cctv_name.split('/')[5].split('_')[0]+"/"+savename #저장되는 경로 
        print(path)

        
        keyword=cctv_name.split('/')[5].split('_')[0] # 저장하고자하는 폴더 
        if not os.path.isdir('CCTV_Video/{}'.format(keyword)): # 폴더가 없으면 생성하는 코드
            os.mkdir('CCTV_Video/{}'.format(keyword))          # ...
        
        urllib.request.urlretrieve(cctv_name, path) #비디오 저장
        print("저장완료")
    


    


while(1):
    for i in range(11,51):
        num = str(i)
        target_url = "http://traffic.daejeon.go.kr/map/trafficInfo/cctvCk.do?cctvId=CCTV"+num #cctv영상 링크
        print(target_url)
        get_eminem_video_link(target_url)
    time.sleep(3600) # 대전 교통정보시스템 기준 2분30초마다 업데이트 , 타임슬립 1시간으로 지정 
