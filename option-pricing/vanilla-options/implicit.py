#%%
###################
# Package Imports #
###################

import numpy as np
from scipy.sparse import diags
from scipy.linalg import solve_banded
import matplotlib.pyplot as plt

from scipy.stats import norm

#%%
#####################
# Option Parameters #
#####################

# American or European?
TYPE = 'American'

# Call or Put?
PUT_F = True

#

S_MAX = 300 # maximum underlying price for grid

K = 100 # strike price

T = 5 # Maturity (years)

SIGMA = 0.25 # underlying volatility

R = 0.045 # risk-free-rate

Q = 0 #0.02 # dividend yield

############################
# Computational Parameters #
############################
N_T = 20000 # Number of time points

N_S = 1000 # Number of underlying stock points

# Vectors

T_VEC = np.linspace(0, T, N_T)

Z_VEC = np.linspace(np.log(0.001), np.log(S_MAX), N_S)

# Delta t and Z terms

dt = T_VEC[1]- T_VEC[0]

dZ = Z_VEC[1]- Z_VEC[0]
######################
# MODEL COEFFICIENTS #
######################

# Implicit #
############

alpha_imp = dt*((R-Q-0.5*SIGMA**2)/(2*dZ) - SIGMA**2/(2*dZ**2))

beta_imp = (1+R*dt+SIGMA**2*dt/dZ**2)

gamma_imp = -dt*((R-Q-0.5*SIGMA**2)/(2*dZ) + SIGMA**2/(2*dZ**2))


#%%

###############################
# STEP 1: Create Grid Surface #
###############################

# Input Parameters
t_surface, z_surface = np.meshgrid(T_VEC, Z_VEC)

# Compute Maturity and Underlying Value curves
# Maturity
mat_surface = T - t_surface
# S
s_surface = np.exp(z_surface)

# Top Row ([0,:]): z = 0
# Bottom Row ([-1,:]): z=log(S_MAX)

# Left Column ([:,0]): t = 0
# Right Column ([:,-1]): t = T

# Price Surface
price_surface = np.zeros((N_S, N_T))

#%%

###################
# STEP 2: Set BCs #
###################

# Terminal Condition (Pay-Off)

if PUT_F:
    price_surface[:,-1] = np.maximum((K-np.exp(z_surface[:,-1])),0.0)

    # Zero S
    price_surface[0,:] = K*np.exp(-R*mat_surface[-1,:])

    # High S
    price_surface[-1,:] = 0

else:
    price_surface[:,-1] = np.maximum((np.exp(z_surface[:,-1])-K),0.0)

    # Zero S
    price_surface[0,:] = 0
    
    # High S
    price_surface[-1,:] = s_surface[-1,:] -K*np.exp(-R*mat_surface[-1,:])


#%%
# Solve the price curve

for t in range(N_T-1,0,-1):


    ################################
    # STEP 4: Implicit solve for v #
    ################################

    #Initialise banded matrix
    ab = np.zeros((3,N_S-2))

    # Upper diagonal
    ab[0,1:] = np.ones(N_S-2-1)*gamma_imp

    # Mid diagonal
    ab[1,:] = np.ones(N_S-2)*beta_imp

    # Lower diagonal
    ab[2,:-1] = np.ones(N_S-2-1)*alpha_imp


    rhs = price_surface[1:-1,t].copy()

    # Incorporate known boundary values
    rhs[0] -= alpha_imp * price_surface[0, t-1]
    rhs[-1] -= gamma_imp * price_surface[-1, t-1]

        
    price_surface[1:-1,t-1] = solve_banded((1,1), ab, rhs)

    ###############################
    # STEP 5: If American option, 
    # take max of intrinsic and expected 
    # option value
    ###############################
    if TYPE=='American':
        price_surface[:,t-1] = np.maximum((K-np.exp(z_surface[:,t-1]))
                                        ,price_surface[:,t-1])

#%%
############################
# STEP 6: Plot the surface #
############################

fig,ax = plt.subplots(subplot_kw={"projection":"3d"})

# Plot the 3D surface
surf = ax.plot_surface(mat_surface, s_surface, price_surface
                       , cmap="viridis", edgecolor="none")

# Customize and display the plot
ax.set_title("Numeric Price Curve")
ax.set_xlabel("Maturity (T)")
ax.set_ylabel("Underlying (S)")
ax.set_zlabel("Option Price")

