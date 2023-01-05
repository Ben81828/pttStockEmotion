import pandas as pd

# 讀取cliwc.dicx，建為DataFrame
def loadCliwctoDf():
    with open('LIWC2015 Dictionary - Chinese (Traditional) (v1.5).dicx', encoding='utf-8') as cliwcD:
        lines = cliwcD.readlines()
        i=0
        for line in lines:
            if i == 0:
                l = line.split('\n')[0].split(',')
                df = pd.DataFrame(columns= l)
            if i > 0:
                l = line.split('\n')[0].split(',')
                l = [ 0 if i=='' else i  for i in l ]
                l[0] = l[0].replace('"','') 
                l[0] = l[0].replace(')','\)')
                l[0] = l[0].replace('(','\(')
                l[0] = l[0].replace('+','\+')
                l[0] = l[0].replace('*','\*')
                df.loc[len(df.index)] = l
            i+=1
        return df

df = loadCliwctoDf()
# 讀取cliwc的table寫到Cliwcloader.csv，以便存取
df.to_csv('Cliwcloader.csv',encoding='UTF-8-Sig')
