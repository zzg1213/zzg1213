import time
now=time.time()
print(now)            #时间戳
print()

obj=time.localtime()
print(obj)            #tm_wday=2,星期三 从0开始算
print(type(obj))      #struct_time对象
print()

obj2=time.localtime(60)    #60秒
print(obj2)
print('年份:',obj2.tm_year)
print()

print(time.ctime())   #时间戳对应的易读的字符串
#日期时间格式化
print(time.strftime('%y-%m-%d %H:%M:%S',time.localtime()))
print(time.strftime('%B %a',time.localtime()))

#字符串转成struct_time
print(time.strptime('2008-08-08','%Y-%m-%d'))

time.sleep(5)
print('helloworld')
