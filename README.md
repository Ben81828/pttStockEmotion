## 這是一個抓取PTT股票版文章回文，並解析其情緒組成的練習(資策會期中專題)
######
期中報告投影片
>https://www.canva.com/design/DAFOpSTQkFw/view
---
### 分析流程說明
>1.ptt股票版爬蟲: 11萬篇貼文
>2.C-LIWC計數情緒詞頻
>3.情緒資料預處理(每天資料往前七天加總)
>>(1)七天內文章的情緒詞頻
>>(2)七天內文章下留言的情緒詞頻
>>(3)七天內所有文章所得的讚、噓、箭頭數
>4.合併其他整體經濟特徵
>>(1)美金/台幣匯率
>>(2)當天大盤收盤價
>>(3)當天大盤漲跌幅
>5.描述統計
>>(1)相關分析
>>(2)漲跌幅300點時的用詞差異
>6.預測隔日漲跌幅
>>全連接神經層預測隔日漲跌幅
>>(後續)加以XGBoost預測
>>>成效欠佳
