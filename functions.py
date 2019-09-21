import numpy as np



def standarize(x):
    x_mean=x.mean()#平均
    std=x.std() #標準偏差
    return (x-x_mean)/std#標準化

def create_matrix(x_std):# tX*w
    return np.vstack([np.ones(x_std.size),
    x_std,
    x_std**2
    ]).T #.Tが転置

def matrix_f(matrix_x,parameter):#a1*b1+12*b2+a3*b3
    return np.dot(matrix_x,parameter)#numpy.dotは内積（一次）、行列積（多次元）

def matrix_E(matrix_x,y,parameter):#parameterはw0,w1,w2
    return 0.5*np.sum((y-matrix_f(matrix_x,parameter))**2)

def polynomial_regression(matrix_x,y):
    parameter=np.random.randint(0,50,3)
    LNR=1e-3#学習率
    defference=1
    count=1#更新回数
    err_before=matrix_E(matrix_x,y,parameter)
    log="({}) parameter:{} error :{:.4f}"
    while defference>1e-2:
        #w0,w1,w2を更新する
        parameter=parameter-LNR*np.dot(matrix_f(matrix_x,parameter)-y,matrix_x)
        err_after=matrix_E(matrix_x,y,parameter)
        defference=err_before-err_after
        err_before=err_after
        if count==1 or count%100==0:
            print(log.format(count,parameter,defference))
        count+=1
    print(log.format(count,parameter,defference))
    return parameter
#データ---
data=np.loadtxt("sales.csv",dtype="int",delimiter=",",skiprows=1)
x=data[:,0]
y=data[:,1]
#-----
"""
a = np.array([33, 44, 54, 23, 25, 55, 32, 76])
ay=standarize(a)
print(ay)
z=create_matrix(ay)

print(z)
"""
x_std=standarize(x)
matrix_x=create_matrix(x_std)
parameter=polynomial_regression(matrix_x,y)
