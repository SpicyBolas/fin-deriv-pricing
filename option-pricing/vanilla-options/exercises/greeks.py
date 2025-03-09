# %%
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Prepare common call-put values
d1 = lambda S0,K,sigma,r,T,q : (np.log(S0/K) + (r-q+sigma**2/2)*T)/(sigma*np.sqrt(T))

d2 = lambda S0,K,sigma,r,T,q : (np.log(S0/K) + (r-q-sigma**2/2)*T)/(sigma*np.sqrt(T))


# Theta intermediates
d1dT = lambda S0,K,sigma,r,T,q : -np.log((S0)/K)*0.5/sigma*(T)**(-3/2) +\
    0.5*(r-q+sigma**2/2)/(sigma*T**0.5)

d2dT = lambda S0,K,sigma,r,T,q : -np.log((S0)/K)*0.5/sigma*(T)**(-3/2) +\
    0.5*(r-q-sigma**2/2)/(sigma*T**0.5)

# Vega intermediates
d1dsig = lambda S0,K,sigma,r,T,q : -(np.log((S0)/K)/(sigma**2*np.sqrt(T)))-((r-q)*np.sqrt(T)/(sigma**2))+np.sqrt(T)/2
d2dsig = lambda S0,K,sigma,r,T,q : d1dsig(S0,K,sigma,r,T,q)-np.sqrt(T)  

def delta(S0,K,sigma,r,T,q,type):
    if type=='call':
        return np.exp(-q*T)*norm.cdf(d1(S0,K,sigma,r,T,q))
    elif type=='put':
        return np.exp(-q*T)*(norm.cdf(d1(S0,K,sigma,r,T,q))-1)

def gamma(S0,K,sigma,r,T,q,type):
    if type=='call':
        return np.exp(-q*T)/(sigma*S0*np.sqrt(T*2*np.pi))*np.exp(-d1(S0,K,sigma,r,T,q)**2/2)
    elif type=='put':
        return np.exp(-q*T)/(sigma*S0*np.sqrt(T*2*np.pi))*np.exp(-d1(S0,K,sigma,r,T,q)**2/2)
    
def theta(S0,K,sigma,r,T,q,type):
    if type=='call':
        return S0*np.exp(-q*T)*(d1dT(S0,K,sigma,r,T,q)/np.sqrt(2*np.pi)*np.exp(-d1(S0,K,sigma,r,T,q)**2/2)-q*norm.cdf(d1(S0,K,sigma,r,T,q)))+\
            -K*np.exp(-r*T)*(d2dT(S0,K,sigma,r,T,q)/np.sqrt(2*np.pi)*np.exp(-d2(S0,K,sigma,r,T,q)**2/2)-r*norm.cdf(d2(S0,K,sigma,r,T,q)))
    elif type=='put':
        return S0*np.exp(-q*T)*(d1dT(S0,K,sigma,r,T,q)/np.sqrt(2*np.pi)*np.exp(-d1(S0,K,sigma,r,T,q)**2/2)+q*norm.cdf(-d1(S0,K,sigma,r,T,q)))+\
            -K*np.exp(-r*T)*(d2dT(S0,K,sigma,r,T,q)/np.sqrt(2*np.pi)*np.exp(-d2(S0,K,sigma,r,T,q)**2/2)+r*norm.cdf(-d2(S0,K,sigma,r,T,q)))
    
def vega(S0,K,sigma,r,T,q,type):
    if type=='call':
        return d1dsig(S0,K,sigma,r,T,q)*((S0)*np.exp(-q*T)/np.sqrt(2*np.pi))*np.exp(-0.5*d1(S0,K,sigma,r,T,q)**2)\
            -d2dsig(S0,K,sigma,r,T,q)*(K*np.exp(-r*T)/np.sqrt(2*np.pi))*np.exp(-0.5*d2(S0,K,sigma,r,T,q)**2) 
    elif type=='put':
        return -d1dsig(S0,K,sigma,r,T,q)*((S0)*np.exp(-q*T)/np.sqrt(2*np.pi))*np.exp(-0.5*d1(S0,K,sigma,r,T,q)**2)\
            +d2dsig(S0,K,sigma,r,T,q)*(K*np.exp(-r*T)/np.sqrt(2*np.pi))*np.exp(-0.5*d2(S0,K,sigma,r,T,q)**2)
    
def rho(S0,K,sigma,r,T,q,type):
    if type=='call':
        return (S0)*np.exp(-q*T)*np.sqrt(T)/sigma/np.sqrt(2*np.pi)*np.exp(-d1(S0,K,sigma,r,T,q)**2/2)\
            +T*K*np.exp(-r*T)*norm.cdf(d2(S0,K,sigma,r,T,q))+\
            -K*np.exp(-r*T)*(np.sqrt(T)/sigma)/np.sqrt(2*np.pi)*np.exp(-d2(S0,K,sigma,r,T,q)**2/2)
    elif type=='put':
        return (S0)*np.exp(-q*T)*np.sqrt(T)/sigma/np.sqrt(2*np.pi)*np.exp(-d1(S0,K,sigma,r,T,q)**2/2)\
            -T*K*np.exp(-r*T)*norm.cdf(-d2(S0,K,sigma,r,T,q))+\
            -K*np.exp(-r*T)*(np.sqrt(T)/sigma)/np.sqrt(2*np.pi)*np.exp(-d2(S0,K,sigma,r,T,q)**2/2)


S0 = 50
K = 48
sigma = 0.25
r = 0.08
T = 1
q=0

S0_vec = np.linspace(0.01,100,1000)
sigma_vec = np.linspace(0.02,0.70,1000)
r_vec = np.linspace(0.02,0.15,500)
T_vec = np.linspace(0.25,10,1000)

#%%
delta_vec = delta(S0_vec,K,sigma,r,T,q,'call')

fig,ax = plt.subplots()
ax.plot(S0_vec,delta_vec)
ax.set_title('Delta (Call)')
ax.set_xlabel('S0')
ax.set_ylabel('Delta')
ax.axvline(K,color='red',linestyle='--')

delta_vec = delta(S0_vec,K,sigma,r,T,q,'put')

fig,ax = plt.subplots()
ax.plot(S0_vec,delta_vec)
ax.set_title('Delta (Put)')
ax.set_xlabel('S0')
ax.set_ylabel('Delta')
ax.axvline(K,color='red',linestyle='--')


#%%
gamma_vec = gamma(S0_vec,K,sigma,r,T,q,'call')

fig,ax = plt.subplots()
ax.plot(S0_vec,gamma_vec)
ax.set_title('Gamma (Call)')
ax.set_xlabel('S0')
ax.set_ylabel('Gamma')
ax.axvline(K,color='red',linestyle='--')
gamma_vec = gamma(S0_vec,K,sigma,r,T,q,'put')

fig,ax = plt.subplots()
ax.plot(S0_vec,gamma_vec)
ax.set_title('Gamma (Put)')
ax.set_xlabel('S0')
ax.set_ylabel('Gamma')
ax.axvline(K,color='red',linestyle='--')

#%%

theta_vec = theta(S0,K,sigma,r,T_vec,q,'call')

fig,ax = plt.subplots()
ax.plot(T_vec,theta_vec)
ax.set_title('Theta (Call)')
ax.set_xlabel('T')
ax.set_ylabel('Theta')

fig,ax = plt.subplots()
ax.plot(S0_vec,theta(S0_vec,K,sigma,r,T,q,'call'))
ax.set_title('Theta (Call)')
ax.set_xlabel('S0')
ax.set_ylabel('Theta')

theta_vec = theta(S0,K,sigma,r,T_vec,q,'put')
fig,ax = plt.subplots()
ax.plot(T_vec,theta_vec)
ax.set_title('Theta (Put)')
ax.set_xlabel('T')
ax.set_ylabel('Theta')

theta_vec = theta(S0_vec,K,sigma,r,T,q,'put')
fig,ax = plt.subplots()
ax.plot(S0_vec,theta_vec)
ax.set_title('Theta (Put)')
ax.set_xlabel('S0')
ax.set_ylabel('Theta')

from matplotlib import cm
from matplotlib.ticker import LinearLocator

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
X, Y = np.meshgrid(S0_vec, T_vec)
Z = theta(X,K,sigma,r,Y,q,'call')

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(0, 5)

# %%
vega_vec = vega(S0,K,sigma_vec,r,T,q,'call')

fig,ax = plt.subplots()
ax.plot(sigma_vec,vega_vec)
ax.set_title('Vega (Call)')
ax.set_xlabel('Sigma')
ax.set_ylabel('Vega')

vega_vec = vega(S0,K,sigma_vec,r,T,q,'put')

fig,ax = plt.subplots()
ax.plot(sigma_vec,vega_vec)
ax.set_title('Vega (Put)')
ax.set_xlabel('Sigma')
ax.set_ylabel('Vega')

vega_vec = vega(S0_vec,K,sigma,r,T,q,'call')

fig,ax = plt.subplots()
ax.plot(S0_vec,vega_vec)
ax.set_title('Vega (Call)')
ax.set_xlabel('S0')
ax.set_ylabel('Vega')

vega_vec = vega(S0_vec,K,sigma,r,T,q,'put')

fig,ax = plt.subplots()
ax.plot(S0_vec,vega_vec)
ax.set_title('Vega (Put)')
ax.set_xlabel('S0')
ax.set_ylabel('Vega')


# %%
rho_vec = rho(S0,K,sigma,r_vec,T,q,'call')

fig,ax = plt.subplots()
ax.plot(r_vec,rho_vec)
ax.set_title('Rho (Call)')
ax.set_xlabel('r')
ax.set_ylabel('Rho')

rho_vec = rho(S0,K,sigma,r_vec,T,q,'put')

fig,ax = plt.subplots()
ax.plot(r_vec,rho_vec)
ax.set_title('Rho (Put)')
ax.set_xlabel('r')
ax.set_ylabel('Rho')

rho_vec = rho(S0_vec,K,sigma,r,T,q,'call')

fig,ax = plt.subplots()
ax.plot(S0_vec,rho_vec)
ax.set_title('Rho (Call)')
ax.set_xlabel('S0')
ax.set_ylabel('Rho')

vega_vec = vega(S0_vec,K,sigma,r,T,q,'put')

fig,ax = plt.subplots()
ax.plot(S0_vec,rho_vec)
ax.set_title('Rho (Put)')
ax.set_xlabel('S0')
ax.set_ylabel('Rho')

# %%
