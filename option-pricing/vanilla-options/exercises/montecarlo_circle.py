#%%
import numpy as np
from scipy.stats import uniform
import matplotlib.pyplot as plt


#%%

#Brute force montecarlo
samples = 1000
pi_list = []

for i in range(samples):
    total = 1000
    circ_count = 0
    for j in range(total):
        points = uniform.rvs(size=2,loc=-0.5,scale=1.0)
        
        r2 = points[0]**2 + points[1]**2

        if r2 <= 1/4:
            circ_count += 4

    pi_est = circ_count/total 
    pi_list.append(pi_est)



# %%
plt.hist(x=pi_list,bins=30)
#%%
# Stratified Sampling

# Uniform distribution between -0.5 and 0.5 for x and y

# Break each into 100 points as a grid 

Nsims = 100

X = np.linspace(-0.5,0.5,Nsims,endpoint=True)

Y = np.linspace(-0.5,0.5,Nsims,endpoint=True)

# initialise circle counter 
circ_count = 0

for x in X:
    for y in Y:
        if x**2 + y**2 <= 1/4:
            circ_count += 4

pi_est = circ_count/(len(X)*len(Y))

print(pi_est)


# %%
