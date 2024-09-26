import numpy as np
import matplotlib.pyplot as plt


x=np.array([3.5,3.69,3.44,3.43,4.34,4.42,2.37])
y=np.array([18,15,18,16,15,14,24])

n=len(x)
neta=0.01

iterations=20000

w=0.0
b=0.0
def predict(x,b,w):
    return w*x+b

def mse(y,y_pred):
    return np.mean((y-y_pred)**2)

def calculate(x,b,w,y,n,iterations,neta):
    
    for i in range(iterations):
        
        y_pred=predict(x,b,w)
        
        mse_=mse(y,y_pred)
        print(f"mse at {i} th time is:{mse_}")
        
        dw= -2/n * np.sum((y-y_pred)*x)
        db= -2/n * np.sum(y-y_pred)
        
        w-=neta*dw
        b-=neta*db
    print(f"weight:{w},bias:{b}")
    return w,b
w,b=calculate(x,b,w,y,n,iterations,neta)

line=w*x+b

plt.scatter(x,y,color='blue',label="data points")
plt.plot(x,line,color='black',label=f"regression line {w}*x+{b}")

plt.xlabel('pounds')

plt.ylabel("miles per gallon")

plt.legend()
plt.show()