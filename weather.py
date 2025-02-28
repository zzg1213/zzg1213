import requests
import re

#定义函数
def get_html():
    url='https://www.weather.com.cn/weather40d/101050101.shtml'
    resp=requests.get(url)
    #设置编码格式
    resp.encoding='utf-8'
    return resp.text

def parse_html(html_str):
    city = re.findall('<span class="name">([\u4e00-\u9fa5]*)</span>', html_str)
    weather = re.findall('<span class="weather">([\u4e00-\u9fa5]*)</span>', html_str)
    wd = re.findall('<span class="wd">(.*)</span>', html_str)
    zs = re.findall('<span class="zs">([\u4e00-\u9fa5]*)</span>', html_str)

    lst=[]
    for a,b,c,d in zip(city,weather,wd,zs):    #打包
        lst.append([a,b,c,d])

    return lst
