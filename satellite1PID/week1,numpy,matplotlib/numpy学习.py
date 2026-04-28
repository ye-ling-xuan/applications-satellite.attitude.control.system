import numpy as np
arr=np.array(5)
print(arr)
print("arr.ndim:",arr.ndim)

arr=np.array([1,2,3,4,5])
print(arr)
print("arr.min:",arr.min())
print("arr.max:",arr.max())

##ndarray的创建
#1.使用array函数
arr=np.array([1,2,3])
print(arr)

#2.使用zeros函数
arr=np.zeros((2,3))
arr=np.ones((3,4))
arr=np.arange(12).reshape((3,4))
print(arr)
arr=np.linspace(1,10,6)
print(arr)

#numpy的基本运算
a=np.array([10,20,30,40])
b=np.arange(4)
c=a+b
print(a,b)
print(c)