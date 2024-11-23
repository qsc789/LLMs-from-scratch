import requests
headers={
    'User_Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36 Edg/130.0.0.0',
    'Cookkie':'_ga=GA1.1.1853750435.1702192352;'
}
url='http://csujwc.its.csu.edu.cn/jsxsd/pyfa/pyfazd_query?Ves632DSdyV=NEW_XSD_PYGL'
responce=requests.get(url)
with open('test1.html','w',encoding='utf-8') as f:
    f.write(responce.text)
responce.close()