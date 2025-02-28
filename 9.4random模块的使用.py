import random
#设置随机数的种子
random.seed(10)                      #interval [0,1)
print(random.random())
print(random.random())
print()

random.seed(10)
print(random.randint(1,100))    #Return random integer in range [a,b]

for i in range(10):                  #[a,b)步长为k
    print(random.randrange(1,10,3),end='')

print(random.uniform(1,100))    #[a,b]随机小数

lst=[i for i in range(1,11)]
print(random.choice(lst))

#随机排序
random.shuffle(lst)
print(lst)

random.shuffle(lst)
print(lst)
