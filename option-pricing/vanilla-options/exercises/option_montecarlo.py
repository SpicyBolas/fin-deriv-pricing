#%%
import numpy as np
import matplotlib.pyplot as plt
# Constants and Initial Conditions #
###################################
S0 = 50
r = 0.10
sigma =0.40
K = 50
T = 5/12

# Parameters #
##############
N = 100 # Number of time nodes
Nsims = 10000 # Number of simulations
# Calculate dt and T vector
dt = T/(N-1)
Tvec = np.linspace(0,T,N)

# Initialise matrix with these dimensions
S_matrix = np.zeros((Nsims,N))

# Set up initial value of S0
S_matrix[:,0] = S0

for i in range(Nsims):
    for j in range(1,N):
        S_matrix[i,j] = S_matrix[i,j-1]*np.exp((r-sigma**2/2)*dt+sigma*np.random.normal(0,1,size=None)*np.sqrt(dt))


#%%
plt.plot(Tvec,S_matrix.T)
plt.xlabel('time (years)')
plt.ylabel('S')
plt.show()


#%%
optionValue = np.mean(np.maximum((S_matrix[:,-1]-K)*np.exp(-r*T),0))
optionErr = np.std(np.maximum((S_matrix[:,-1]-K)*np.exp(-r*T),0),ddof=1)/np.sqrt(Nsims)

CI = [optionValue-1.96*optionErr,optionValue+1.96*optionErr]

print(f'Option Value is: {optionValue} with std error: {optionErr}')

print(f'95% Confidence Interval between: [{CI[0]},{CI[1]}]')
# %%
