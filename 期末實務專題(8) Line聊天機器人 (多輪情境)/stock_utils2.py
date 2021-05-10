from datetime import datetime, timedelta
import time
import pandas as pd
import requests, json
import yfinance as yf
import re


#抓取儲存股票名稱
def get_stock_name():
    # current date and time
    company2symbol = {}
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    filters = ['0'+str(i) for i in range(1,10)]
    filters.extend([str(i) for i in range(10,32)])

    for filter in filters:

        url = 'https://www.twse.com.tw/zh/api/codeFilters?'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Mobile Safari/537.36',
            'Referer': 'https://www.twse.com.tw/zh/page/trading/exchange/STOCK_DAY.html'
        }
        params = {
            'filter': str(filter),
            '_': timestamp
        }
        res = requests.get(url,headers=headers,params=params)
        jdatas = json.loads(res.text)['resualt']
        for data in jdatas:
            symbol = data.split('\t')[0]
            company = data.split('\t')[1]
            company2symbol[company] = symbol
       
        time.sleep(1)
        
    return company2symbol

def save_company2symbol(company2symbol,path):
    df_symbol = pd.DataFrame.from_dict(company2symbol,orient='index')
    df_symbol.columns = ['symbol']
    df_symbol.to_csv(path)
    return df_symbol
    
#抓取股票資料
def get_stock_price(df_symbol,query,start,end):   
    symbol = df_symbol.loc[query].symbol+'.TW'
    data = yf.download(symbol, start=start, end=end)
    return data

#選擇資料
def get_selected_info(data,columns,query,symbol_name):
    message=f'{symbol_name} {query} \n'
    for idx, row in data.iterrows():
        message += f'*{idx.date()}    {row[columns[query]]} \n'
    return message
    
if __name__ == '__main__':
  
