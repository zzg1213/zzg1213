import requests

url='https://hitgs.hit.edu.cn/_upload/tpl/03/b7/951/template951/images/logo.svg'
resp=requests.get(url)

#保存到本地 文件读写
with open('logo.svg','wb') as file:               #wb写为二进制
    file.write(resp.content)
