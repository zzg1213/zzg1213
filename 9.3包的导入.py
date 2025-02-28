import admin.my_admin as a       #包名.模块名
a.info()                         #__init__.py中的内容被自动执行,只执行一次
print()

from admin import my_admin as b  #from 包名 import 模块名 as 别名
b.info()
print()

from admin.my_admin import info  #from 包名.模块名 import 函数/变量等
info()
print()

from admin.my_admin import *     #from 包名.模块名 import *
print(name)
