#
import name_introduce
print(name_introduce.name)
name_introduce.fun()

import name_introduce as a
print(a.name)
a.fun()

#
from name_introduce import name    #导入变量
print(name)

from name_introduce import fun     #导入函数
fun()

from name_introduce import  *      #通配符
print(name)
fun()

import math,time,random
