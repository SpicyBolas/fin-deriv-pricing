# %%
import numpy as np
import matplotlib.pyplot as plt

mu = 0.15
sigma = 0.3

S0 = 100
T = 1
dt = 1/52
N = round(T/dt + 1)
nsims = 1

def stock_mc(S,mu,sigma):
    S_new = mu*S*dt+sigma*S*np.random.normal(0,1,size=None)*np.sqrt(dt)+S
    return S_new

T_vec = np.arange(0,T+dt,dt)

S_vec = np.zeros((N,nsims))

for idx in range(nsims):
    S_vec[0,idx] = S0

for sim in range(nsims):
    for idx in range(1,N):
        S_vec[idx,sim] = stock_mc(S_vec[idx-1,sim],mu,sigma)
        

fig,ax = plt.subplots()
ax.plot(T_vec,S_vec)

ax.set_xlabel('Time (Years)')
ax.set_ylabel('Stock Value')

fig.show()
# %%
