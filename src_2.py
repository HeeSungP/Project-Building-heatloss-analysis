from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.alert import Alert
import time
import re
from bs4 import BeautifulSoup


# 유의점


## 크롬 웹 드라이버

#### 1. 크롬 웹 드라이버 버전에 맞게 py파일과 같은 path에 다운로드 해야함.
#### 2. 크롬 업데이트시 크롬 웹 드라이버도 버전에 맞게 업데이트 해야함.
#### 3. 크롬 버전은 설정 - Chrome 정보에서 확인 가능
#### 4. https://chromedriver.chromium.org/downloads (드라이버 다운로드 위치)



## 지도 사용시

#### 1. 건물 이름 등으로 검색 시 검색이 잘 이루어지지 않을 수 있음
#### 2. 정확한 주소로 검색하는 방법 추천
#### 3. 검색이 잘 이루어지지 않을 경우 띄워진 크롬 창 내에서 다시 검색해서 사용 가능



## 추가사항

#### 1. 개인적으로는 검색어를 python에서 입력하는 방법 말고 크롬창 내에서 입력하는 방식이 더 유저 친화적으로 작동할 것으로 생각함





# Google Map
def size_measure_google():
    path = './chromedriver.exe'
    options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(path, options=options) 
    url_base = 'https://www.google.co.kr/maps/place/'
    driver.get(url_base)
    
    while True:
        try:
            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            ruler = soup.find(id='ruler')
            p =re.compile(r'총 거리: .*span .*>(.*)m</span>')
            m = p.search(str(ruler))
            length = m.group(1)

        except:
            try:
                html = driver.page_source
                soup = BeautifulSoup(html, 'html.parser')
                open_check = soup.find('head')
            except:
                break
    #거리 측정이 잘 되지 않았을 경우 에러 메세지 출력
    try:
        return length
    except:
        return print('거리 측정에 실패했습니다. 다시 시도해주세요. ')
    
# Naver Map
def size_measure_naver():
    path = './chromedriver.exe'
    options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(path, options=options) 
    url_base = 'https://map.naver.com/'
    driver.get(url_base)

    while True:
        try:
            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            ruler = soup.find(class_='toolbox_box distance')
            length = ruler.text
        except:
            try:
                html = driver.page_source
                soup = BeautifulSoup(html, 'html.parser')
                open_check = soup.find('head')
            except:
                break
    #거리 측정이 잘 되지 않았을 경우 에러 메세지 출력
    try:
        return length
    except:
        return print('거리 측정에 실패했습니다. 다시 시도해주세요. ')