import numpy as np
from copy import deepcopy

# Number of simulations
NSims = 1000
#Number of time steps
N = 1000

# Constants
T=9/12
dt = T/N

S0 = 400
r = 0.06
q = 0.04
sigma = 0.2
payOffVec = []

for i in range(NSims):
    Smin = deepcopy(S0)
    S = deepcopy(S0)
    for j in range(N):
        S = S*np.exp((r-q+sigma**2/2)*dt+sigma*np.random.standard_normal()*np.sqrt(dt))
        if S < Smin:
            Smin = deepcopy(S)
    
    payOffVec.append(max(S-Smin,0))

mPayOff = np.mean(payOffVec)

stdPayOff = np.std(payOffVec,ddof=1)

Err = stdPayOff/np.sqrt(NSims)

print(f'Value of {mPayOff} with standard error: {Err}')

print(f'95% CI value = {1.96*Err}')
