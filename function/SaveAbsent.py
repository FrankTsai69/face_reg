import pandas as pd
import datetime
import os
'''example:
    let col =['name','SID','SOD','state'];
    let data ={"2024-9-4":{
        "Frank":{"SID":"2024/9/4 08:20","SOD":"2024/9/4 16:38","state":"late"},
        "Amy":{"SID":"2024/9/4 08:09","SOD":"2024/9/4 16:30","state":"excused"},
        "Ben":{"SID":"2024/9/4 08:07","SOD":"2024/9/4 16:38","state":"present"},
        "Mary":{"SID":"2024/9/4 08:20","SOD":"2024/9/4 16:38","state":"present"},
    }};

    let date ="2024-9-4";
'''

def savedata():
    with open("./data.js",'w')as t:
        t.write(data)
    
def main(date,name):
    today=str(datetime.date.today())
    path=f"date{today}.pkl"
    plk={}
    if path in os.listdir('./absent/date'):
        plk=pd.read_pickle(f'./absent/date/{path}')
    else:
        plk[today]={name:[]}
    if name in plk[today].keys():
        plk[today][name].append(date)
    else: plk[today][name]=[date]
    #print(plk)
    pd.to_pickle(plk,f'./absent/date/{path}')
        
    
if __name__ == "__main__":

    data={}
    absent={'date':{'default':['datelist']}}
    #pd.to_pickle(plk,'../absent/date.plk')
        
