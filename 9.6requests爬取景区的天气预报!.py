import  requests
import re

url='https://www.weather.com.cn/weather40d/101050101.shtml'
resp=requests.get(url)
#设置编码格式
resp.encoding='utf-8'
print(resp.text)

'''
<span class="name">三亚</span>
<span class="weather">多云</span>
<span class="wd">29/20℃</span>
<span class="zs">适宜</span>
'''

city=re.findall('<span class="name">([\u4e00-\u9fa5]*)</span>',resp.text)
weather=re.findall('<span class="weather">([\u4e00-\u9fa5]*)</span>',resp.text)
wd=re.findall('<span class="wd">(.*)</span>',resp.text)
zs=re.findall('<span class="zs">([\u4e00-\u9fa5]*)</span>',resp.text)
print(city)
print(weather)
print(wd)
print(zs)

lst=[]
for a,b,c,d in zip(city,weather,wd,zs):    #打包
    lst.append([a,b,c,d])
print(lst)
print()
for item in lst:
    print(item)
