import math
import numpy as np
from scipy import integrate
import scipy.special as special
from scipy.misc import derivative


def fd_Caputo(f,t,a,nu,h):
    x2 = lambda s: (1.0/special.gamma(1-nu))*derivative(f, s, dx=h)/(t-s)**nu
    return integrate.quad(x2, a, t)


def WIntegral(f,t,a,nu):
    x2 = lambda s: (1.0/special.gamma(1-nu))*f(t)/(t-s)**nu
    return integrate.quad(x2, a, t)


def fd_RiemannLiouville(f,t,a,nu,h):
    def fintegral(t):
        return WIntegral(f,t,a,nu)[0]
    return derivative(fintegral, t, dx=h)


def RL_integral(f,a,t,alpha):
    x2 = lambda s: ((t-s)**(alpha-1))*f(s)/special.gamma(alpha)
    return integrate.quad(x2, a, t)[0]


def FODE(f,Initial,Interv,dx,alpha):
    m = math.ceil(alpha)
    
    if len(Initial) != m:
        print("The number of initial conditions is wrong!")
    
    #discretization of independent variable
    N_ = int((Interv[1]-Interv[0])/dx)
    x = np.linspace(Interv[0],Interv[1],N_+1)
    #Initial setup for dependent variable
    y = np.zeros(len(x))
    y[0] = Initial[0]
    
    def b(j,n):
        return ((n-j)**alpha - (n - 1 - j)**alpha)/special.gamma(alpha+1)
    
    def a(j,n):
        factor = (dx**alpha)/special.gamma(alpha + 2)
        if j == 0:
            return factor*((n-1)**(alpha+1) - (n-1-alpha)*(n**alpha))
        elif (1 <= j) and (j <= n-1):
            return factor*((n-j+1)**(alpha+1) - 2*(n-j)**(alpha+1) + (n-1-j)**(alpha+1))
        elif j == n:
            return factor
        else:
            print("Something went wrong in calculation for a(j,n)")
    
    #Calculation of the Numerical Solution
    for n in range(0,N_):
    
        #Predictor u_{n+1}^P
        sum1 = 0
        for j in range(0,m):
            sum1 += (x[n+1]**j)*Initial[j]/math.factorial(j)
        sum2 = 0
        for j in range(0,n+1):
            sum2 += (dx**alpha)*b(j,n+1)*f(x[j],y[j])
            #print("f(x[j],y[j]) = ",f(x[j],y[j]))
        uP = sum1 + sum2
    
        #Approximation for u_{n+1}
        sum3 = 0
        for j in range(0,n+1):
            sum3 += a(j,n+1)*f(x[j],y[j])
            #print("a(j,n+1) = ",a(j,n+1))
        
        #y[n+1] = sum1 + sum2 #Euler
        #y[n+1] = sum1 + sum3 #Adams (upper limit n -> n+1 ?)
        y[n+1] = sum1 + sum3 + a(n+1,n+1)*f(x[n+1],uP) #Adams Predictor-Corrector
        #print("sum1 = ", sum1)
        #print("sum2 = ", sum2)
        #print("sum3 = ", sum3)
    return (x,y)


""" 
Comments:
i) To decide which method for integration (or several) to use. quad seems to be the most
primitive and doesn't deal appropriately for oscillating functions
ii) To use the same notation: We are using left operators and the order a,t instead of
t, a seems more natural
"""