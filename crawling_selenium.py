from selenium import webdriver    
# selenium으로 키를 조작하기 위한 import
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.common.exceptions import WebDriverException

import json
import re
import json
import requests
import time


###Description
# #동작: 자동으로 사이트 같은거 들어가는 코드로 짬.
# 1. 처음에 google scholar페이지로 이동.
# 2. 검색창에 내가 원하는 키워드 침.
# 3. 첫번째 검색결과 부터 차례대로 들어가서 데이터 긁어옴.
# 4. 다 읽었으면 다음 페이지 넘어감 (max_page 까지)
# 5. json으로 저장


# chromedriver
# 압축해제한 웹드라이버의 경로와 파일명 지정
# driver = webdriver.Chrome('./chromedriver')
driver = webdriver.Chrome() 

# Google Scholar 페이지로 이동
driver.get("https://scholar.google.com/")

# 검색어 입력
search_query = "dialogue system"  # 검색할 내용을 입력하세요
search_box = driver.find_element(By.ID, "gs_hdr_tsi")
search_box.send_keys(search_query)
search_box.submit()

# 검색 결과 페이지가 로드되기를 기다림 (필요에 따라 시간 조절)
# 이 부분은 화면이 완전히 로드될 때까지 기다릴 수 있도록 수정해야 합니다.
driver.implicitly_wait(10)

# 검색 결과에서 논문 제목과 URL 추출 (최대 10개)
results = driver.find_elements(By.CLASS_NAME, "gs_ri")


page_number = 1

#최대 몇페이지 까지 넘어갈건지 (한 페이지 약 10개 논문 정도 있는 것 같음)
max_pages = 3

paper_info_list = []

while page_number <= max_pages:
    print(f"Page {page_number}:")
    results = driver.find_elements(By.CLASS_NAME, "gs_ri")
    
    for result in results:
        title_element = result.find_element(By.TAG_NAME, "h3")
        title = title_element.text
        url_element = result.find_element(By.TAG_NAME, "a")
        url = url_element.get_attribute("href")
        
        paper_info_list.append([title, url])
        print("Title:", title)
        print("URL:", url)
        print()
    
    # 다음 페이지로 이동
    next_page_button = driver.find_element(By.ID, "gs_n")
    next_page_button.click()

    page_number += 1

# Load Page
# chrome을 띄워 네이버 블로그 페이지를 연다.

# 현재 URL을 출력
print(driver.current_url)

abstracts_list = []
p_num = 0
for title, url in paper_info_list[:]:

    #여기에서 논문 뽑을 사이트 정해야함. -> 나는 ieee, arxiv, sciencedirect 뽑음, URL에서 보고 저 단어 있는 페이지만 들어가서 데이터 긁어옴
    if 'emnlp' in url:
        driver.get(url)
        meta_element = driver.find_element(By.CSS_SELECTOR, 'meta[property="twitter:description"]') # meta[property="twitter:description"] 이부분은 학술지마다 달라서 찾아봐야됨. 나한테 어디거 쓸건지 말하면 찾아줌 
        # "content" 속성 값 가져오기
        description_content = meta_element.get_attribute("content")
    
    elif 'naacl' in url:
        driver.get(url)
        meta_element = driver.find_element(By.CSS_SELECTOR, 'meta[property="og:description"]')
        # "content" 속성 값 가져오기
        description_content = meta_element.get_attribute("content")
        
    elif 'acl' in url:
        driver.get(url)
        abstract_element = driver.find_element(By.XPATH, "//div[contains(@class, 'abstract')]")
        description_content = abstract_element.text

        parts = description_content.split("\n")
        description_content = "\n".join(parts[1:])

    else: # 위에 세군데 학술지 아닌 곳은 안들어가고 넘김
        continue
    
    print("Title : ", title)
    print("URL: ", url)
    print("Abstract:")
    print(description_content)
    abstracts_list.append([title, url, description_content])
        
    print('\n')
    
    p_num += 1
    
    if p_num > 101:
        break


    
# 텍스트 파일에 저장
# with open('abstracts.txt', 'w') as file:
#     for item in abstracts_list:
#         title, url, content = item
#         file.write(f"{title}\n{url}\n{content}\n\n")
        
abstracts_dict_list = [{'title': item[0], 'url': item[1], 'content': item[2]} for item in abstracts_list]
with open('abstracts.json', 'w') as file:
    json.dump(abstracts_dict_list, file, indent=4)

driver.close()
