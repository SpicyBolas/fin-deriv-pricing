##########
# Calculates the binomial correlation measure
#  for two companies

#%%
# import required packages
import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal

# Define constants
Qa = 0.2 # Company A default probability in 2 years
Qb = 0.15 # Company A default probability in 2 years

# Define the covariance matrix input
# Note this is the same as the correlation matrix since 
# both std_devs are 1.
corMat = np.array([[1,0.3],[0.3,1]])


#%%

# Calculate binomial correlation measure

# quantiles 
qa = norm.ppf(Qa)
qb = norm.ppf(Qb)

beta = (multivariate_normal(mean=[0,0],cov=corMat).cdf([qa,qb])-Qa*Qb)/\
    np.sqrt((Qa-Qa**2)*(Qb-Qb**2))


print(beta)
# %%
