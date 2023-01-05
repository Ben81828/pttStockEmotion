##記得改out.txt日期
##記得改PttStock_CliwcParsed.csv日期
import pandas as pd
import re

# 讀取Cliwcloader.csv，建為DataFrame
def loadCliwc():
    cliwcDF = pd.read_csv('Cliwcloader.csv', encoding='UTF-8-Sig')
    return cliwcDF

# 讀取out.txt，輸出以列為元素的list
def loadOut():
    with open('out1008.txt', encoding='UTF-8-Sig') as out:
        lines = out.readlines()
    return lines

df = loadCliwc().iloc[:,1:]      # 把cliwc建好dataFrame，名稱為df。
EntryTup = tuple(df['Entry'])  # 將關鍵字的欄位取出為series，名稱為Entry
lines = loadOut()              # 把out.txt讀出，放入變數lines。(list型態，元素為字串(待轉為字典))
articles = None                #將最後結果articles(DataFrame)先建好，待以迴圈將資料從lines取出，解析完之後一列一列塞資料進artcles

c = 0                          #計次器: 代表跑到第幾筆資料
#迴圈開始解析資料
for line in lines:
    # 將字串從lines取出，轉型為字典，命名為dic
    dic = eval(line)
    # 幫字典新增七個key，分別為'留言推'、'留言噓'、'留言→'、'年'、'月'、'日'、'時'
    dic['留言推'], dic['留言噓'], dic['留言→'],dic['年'],dic['月'],dic['日'],dic['時'] = 0,0,0,0,0,0,0
    time = dic['日期']
    timepattern='(\S*)\s*(\S*)\s*(\d*)\s*(\d*:\d*:\d*)\s*(\d*)'
    date = re.findall(timepattern, time)
    dic['年'],dic['月'],dic['日'],dic['時'] = date[0][-1],date[0][1],date[0][2],date[0][3]
    #計算留言的推數、噓數、箭頭數。並將文章留言放入str1，文章內容放入str2
    message = '' #留言
    content = dic['內容']
    for tup in dic['留言']:
        if tup[0] == '推':
            dic['留言推'] += 1
        elif tup[0] == '噓':
            dic['留言噓'] += 1
        elif tup[0] == '→':
            dic['留言→'] += 1
        else:
            continue
        message += tup[2] #留言內容放到message
        
    # 開始解析message(留言)和content(文章內容)
    # 先解析message
    count1 = [] #建立list，當有關鍵字Entry被解析到時，就會在該Entry的位置計數
    mask1 = []  #建立遮罩list，當有關鍵字Entry被解析到時，就會在該Entry的位置給True，否則False
    for i in range(len(EntryTup)):
        l = [len(re.findall(r'{}'.format(EntryTup[i]), message))]
        mask1 += [True] if l!=[0] else [False]
        count1 += l
    # 把關鍵字位置計數的list轉為series，並只取出計數大於0的部分。
    count1 = pd.Series(count1)
    count1 = count1[count1>0]
    # 利用遮罩，把剛剛有解析到的每個關鍵字所對應到的liwc特徵加權，從df中取出，型態為DataFrame
    parse1 = df[mask1].loc[:,'function':].astype(int)
    # count為計數X關鍵字的向量，parse為關鍵字X特徵的矩陣。count和parse取dot，得計數X特徵的向量，再轉為DataFrame，命名為message_parse
    message_parse = count1.dot(parse1)
    message_parse = pd.DataFrame(message_parse).transpose()
    # 把DataFrame的column中的特徵名稱，加上'message_'
    message_parse.columns = 'message_'+ message_parse.columns
    # 再解析str2
    count2 = [] #建立list，當有關鍵字Entry被解析到時，就會在該Entry的位置計數
    mask2 = []  #建立遮罩list，當有關鍵字Entry被解析到時，就會在該Entry的位置給True，否則False
    for i in range(len(EntryTup)):
        l = [len(re.findall(r'{}'.format(EntryTup[i]), content))]
        mask2 += [True] if l!=[0] else [False]
        count2 += l
    # 把關鍵字位置計數的list轉為series，並只取出計數大於0的部分。
    count2 = pd.Series(count2)
    count2 = count2[count2>0]
    # 利用遮罩，把剛剛有解析到的每個關鍵字所對應到的liwc特徵加權，從df中取出，型態為DataFrame
    parse2 = df[mask2].loc[:,'function':].astype(int)
    # count為計數X關鍵字的向量，parse為關鍵字X特徵的矩陣。count和parse取dot，得計數X特徵的向量，再轉為DataFrame，命名為message_parse
    content_parse = count2.dot(parse2)
    content_parse = pd.DataFrame(content_parse).transpose()
    # 把DataFrame的column中的特徵名稱，加上'content_'
    content_parse.columns = 'content_'+ content_parse.columns
    # 把文章的日期、作者、標題取出，並與content_parse、message_parse合併。得到單筆article資料
    article_info = pd.DataFrame([dic['年'],dic['月'],dic['日'],dic['時'],dic['作者'],dic['標題'],dic['留言推'],dic['留言噓'],dic['留言→']],index=['年','月','日','時','作者','標題','推數','噓數','→數']).transpose()
    article = pd.concat([article_info,content_parse,message_parse],axis=1)
    articles = pd.concat([articles ,article],axis=0)
    # debug欄位
    c+=1
    print('已塞入第 ',c,'筆資料')

#將結果寫入PttStock_CliwcParsed.csv
articles.to_csv('PttStock_CliwcParsed1008.csv',encoding='utf_8_sig',index=False)