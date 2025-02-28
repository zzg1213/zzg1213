from name_introduce import  *
from introduce import  *

fun()    #导入模块具有同名的变量和函数,后导入的会将之前导入的进行覆盖
print()

#不想覆盖,解决方案可以使用import
import name_introduce
import introduce
name_introduce.fun()
introduce.fun()
