import numpy as np

a = np.array([33, 44, 54, 23, 25, 55, 32, 76])

def standarize(x):
    x_mean=x.mean()#平均
    std=x.std() #標準偏差
    return (x-x_mean)/std#標準化

y=standarize(a)
print(y)
