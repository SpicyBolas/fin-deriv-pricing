import numpy as np
from copy import deepcopy

# Number of simulations
NSims = 1000
#Number of time steps
N = 1000

# Constants
T=6/12
dt = T/N

S0 = 30
K=30
r = 0.05
q = 0.0
sigma = 0.3
payOffVec = []

for i in range(NSims):
    Smin = deepcopy(S0)
    S = deepcopy(S0)
    S_vec = [S0]

    for j in range(N):
        S = S*np.exp((r-q+sigma**2/2)*dt+sigma*np.random.standard_normal()*np.sqrt(dt))
        S_vec.append(S)        
                

    payOffVec.append(max(np.mean(S_vec)-K,0))

mPayOff = np.mean(payOffVec)

stdPayOff = np.std(payOffVec,ddof=1)

Err = stdPayOff/np.sqrt(NSims)
print('###MONTE CARLO###')
print(f'Value of {mPayOff} with standard error: {Err}')

print(f'95% CI value = {1.96*Err}')

# Repeat using the analytical formula

# Calculate M1
M1 = (np.exp((r-q)*T)-1)*S0/((r-q)*T)

# Calculate M2

M2 = (2*np.exp((2*(r-q)+sigma**2)*T)*S0**2)/\
    ((r-q+sigma**2)*(2*r-2*q+sigma**2)*T**2)+\
        (2*S0**2/((r-q)*T**2))*(1/(2*(r-q)+sigma**2)-np.exp((r-q)*T)/(r-q+sigma**2))


# Use Black's Model to calculate the value of the Asian option
sigma_b = np.sqrt((1/T)*np.log(M2/(M1**2)))

d1 = (np.log(M1/K)+T*sigma_b**2/2)/(np.sqrt(T)*sigma_b)

d2 = (np.log(M1/K)-T*sigma_b**2/2)/(np.sqrt(T)*sigma_b)

from scipy.stats import norm

value = M1*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)

print('###ANALYTICAL FORMULA###')
print(f'Value of {value}')
