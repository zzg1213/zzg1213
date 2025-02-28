import requests

url='https://www.baidu.com/img/PCtm_d9c8750bed0b3c7d089fa7d55720d6cf.png'
resp=requests.get(url)

#保存到本地 文件读写
with open('logo.png','wb') as file:               #wb写为二进制
    file.write(resp.content)
