# import urllib3.request as req
import requests 
from random import randint
from time import sleep
import bs4
import re
import winsound
import logging

def getData(pageurl):
    # 預先建立要插入"上一頁url"的列表
    prePageUrl = []
    prePageUrl = [pageurl] if prePageUrl == [] else prePageUrl
    head0 = {"authority": "www.ptt.cc",
"method": "GET",
"scheme": "https",
"referer": "https://www.ptt.cc/bbs/index.html",
"sec-ch-ua-mobile": "?0",
"sec-ch-ua-platform": "Windows",
"sec-fetch-dest": "document",
"sec-fetch-mode": "navigate",
"sec-fetch-site": "same-origin",
"sec-fetch-user": "?1",
"upgrade-insecure-requests": "1",
"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36",
}
    #記得改日期
    path = 'out1008.txt'
    with open(path, 'a',encoding="utf-8") as f:
        # index計數器
        index = 0
        while True:
            # try:
            indexUrl = prePageUrl[0]
            # 計數器+1並印出
            index += 1
            print("準備讀取並解析頁面",index," : ",indexUrl,end="") 
            # 建立要求連進indexUrl的request_i
            head1=head0
            try:
                request_i  = requests.get(indexUrl, headers = head1)
            except:
                print("handshake failure")
                delay = 100
                print("準備睡眠 " + str(delay) + " 秒")
                sleep(delay)   
                request_i  = requests.get(indexUrl, headers = head1)

            ##print("第",index,"連index頁 ",head1["referer"])
            ###request_i = req.Request(indexUrl, headers=head1)
            #傳送request_i物件，向網站表示要連上url
            ###with req.urlopen(request_i) as response:
                # 指定在讀取網頁資料時，以utf8解碼，將讀取的資料放到data變數
            
            ###articleData=response.read().decode("utf-8")
            # root為解析index頁的資料
            root = bs4.BeautifulSoup(request_i.text, "html.parser")
            print("---已進入頁面")
            # 從root中，找出第一個class為"btn-group btn-group-paging"的div標籤
            div_btn_group_paging = root.find("div", class_="btn-group btn-group-paging")
            # 從class為"btn-group btn-group-paging"的div標籤中，取出所有a標籤
            div_btn_group_atags = div_btn_group_paging.find_all("a")
            # 從所有a標籤中，找出innerHTML="‹ 上頁"的a標籤，並將該a標籤的href屬性取出，並在前面加上"https://www.ptt.cc/"
            prePageUrl = ["https://www.ptt.cc/"+div_btn_group_atag.get('href') for div_btn_group_atag in div_btn_group_atags if div_btn_group_atag.string == "‹ 上頁"]
            

            # 從root中，找出第一個class為"title"的div標籤
            divs_title = root.find_all("div", class_="title")
            # 從所有class為"title"的div標籤中取出a標籤，並將該a標籤的href屬性取出，並在前面加上"https://www.ptt.cc/"
            # 將每個取出的文章url加入到articlesUrlList
            articlesUrlList = [ "https://www.ptt.cc/"+div.a.get('href') for div in divs_title if div.a!=None ]
            

            # 建立寫出文章的計數器
            count = 0
            for articleUrl in articlesUrlList:
                head1["referer"]=indexUrl
                count+=1
                print("準備解析目錄第",index,"頁 : 第",count,"篇文章",end="")

                
                URL = articleUrl
                # 將請求(設定Header與Cookie)request物件並發出
                ##print("進文章",head1["referer"])
                try:
                    request_article = requests.get(URL, headers = head1)
                except:
                    print("handshake failure")
                    for _ in range(2):
                        delay =100
                        print("準備睡眠 " + str(delay) + " 秒")
                        sleep(delay)   
                        request_article = requests.get(URL, headers = head1)
                
                ###request_article = req.Request(URL, headers=head1)
                #傳送request物件，向網站表示要連上url
                
                ####with req.urlopen(request_article) as response:
                
                #指定在讀取網頁資料時，以utf8解碼，將讀取的資料放到data變數
                ####articleData=response.read().decode("utf-8")
                        
                #  把網頁程式碼(HTML) 丟入 bs4模組分析
                root_aritcle = bs4.BeautifulSoup(request_article.text,"html.parser")
                print("---已進入頁面")
                ## PTT 上方4個欄位
                header = root_aritcle.find_all('span','article-meta-value')
                if (header == [])or(len(header)<4):
                    print(header)
                    _ = print("無作者、看板、標題、日期資訊") if header == [] else print("作者、看板、標題、日期有缺")
                    continue
                # 作者
                author = header[0].text
                # 看版
                board = header[1].text
                # 標題
                title = header[2].text
                # 日期
                date = header[3].text


                ## 查找所有html 元素 抓出內容
                main_container = root_aritcle.find(id='main-container')
                # 把所有文字都抓出來，把整個內容切割透過 "-- " 切割成2個陣列
                all_text = main_container.text.split('--')
                # 第0個包含作者、看板、標題、日期、內容
                pre_text = all_text[0]
                # 第-1個包含推噓、留言者、留言內容、留言時間
                after_text = all_text[-1]

                    
                # 把每段文字 根據 '\n' 切開
                texts_0 = pre_text.split('\n')
                texts_last = after_text.split('\n')

                # 內容
                contents = texts_0[2:]
                content = '\n'.join(contents)

                # 留言
                postmessages = []
                for i in range(len(texts_last)):
                    if texts_last[i] == "":
                        continue
                    if texts_last[i][0] == "推" or texts_last[i][0]=="→" or texts_last[i][0]=="噓":
                        postmessages = texts_last[i:-3]
                        break
                # 解析留言內容，放入postmessagesList
                pattern = r'(^["推","→","噓"])\s*(\S*)\s*:\s*(\S*[\s?\S?]*\S)\s*(\d\d/\d\d)\s*(\d\d:\d\d)$'
                postmessagesList = []
                for message in postmessages:
                    _ = re.findall(pattern, message)
                    postmessagesList += _

                #文章內容相關變數放入字典
                dict={'作者':author,'看板':board,'標題':title,'日期':date,'內容':content,'留言':postmessagesList}
        
                line = str(dict)+"\n"
                f.writelines(line)
                #睡眠
                # delay = randint(1,5)
                # print("準備睡眠 " + str(delay) + " 秒")
                # sleep(delay)

                head1["referer"]=URL
                    # 將請求(設定Header與Cookie)request物件並發出
                ##print("出文章: ",head1["referer"])
                try:
                    request_article = requests.get(indexUrl, headers = head1)
                except:
                    print("handshake failure")
                    for _ in range(2):
                        delay =100
                        print("準備睡眠 " + str(delay) + " 秒")
                        sleep(delay)   
                        request_article = requests.get(indexUrl, headers = head1)

                #傳送request物件，向網站表示要連上url
            head0["referer"]=indexUrl
            print("---回目錄頁面")
                        
        # except:
        #     if KeyboardInterrupt:
        #         break
        #     else:
        #         print("錯誤發生")
        #         winsound(2000,500)
        #         continue

pageurl="https://www.ptt.cc/bbs/Stock/index.html"
getData(pageurl)

