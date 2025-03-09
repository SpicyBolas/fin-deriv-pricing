# %%
import numpy as np
import matplotlib.pyplot as plt

a = 20
b = 30

x0 = 50
T = 1
N = 1001
nsims = 1000

dt = T/(N-1)

def wiener_sim(x,a,b):
    x_new = a*dt+b*np.random.normal(0,1,size=None)*np.sqrt(dt)+x
    return x_new

T_vec = np.arange(0,T+dt,dt)

x_vec = np.zeros((N,nsims))

for idx in range(nsims):
    x_vec[0,idx] = x0

for sim in range(nsims):
    for idx in range(1,N):
        x_vec[idx,sim] = wiener_sim(x_vec[idx-1,sim],a,b)
        

fig,ax = plt.subplots()
ax.plot(T_vec,x_vec)

ax.set_ylim([-200,200])

ax.set_xlabel('Time (Years)')
ax.set_ylabel('Stock Value')

fig.show()
# %%
