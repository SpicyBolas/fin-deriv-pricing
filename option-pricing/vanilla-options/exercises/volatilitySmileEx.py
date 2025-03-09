#%%
from binTree import BinomOptionVal
import numpy as np
import pandas as pd

S0 = 75
r = 0.06
sigma =0.25
K_vec = np.arange(30,90,10) #Vector of strike prices
#K = 0.59
T = 0.5

# Create empty lists for high call and put
c_list_high = []
p_list_high = []  

# Obtain values for optimistic case
for K in K_vec:
    option_valuer = BinomOptionVal(S0,K,T,r,sigma,option_type='call',American=False,underlying='S',q=0)
    c = option_valuer.value(1000)
    
    c_list_high.append(c)
    
    option_valuer = BinomOptionVal(S0,K,T,r,sigma,option_type='put',American=False,underlying='S',q=0)
    p = option_valuer.value(1000)

    p_list_high.append(p)

# Pessimistic Case
S0 = 50
sigma = 0.4

# Create empty lists for high call and put
c_list_low = []
p_list_low = []  

# Obtain values for optimistic case
for K in K_vec:
    option_valuer = BinomOptionVal(S0,K,T,r,sigma,option_type='call',American=False,underlying='S',q=0)
    c = option_valuer.value(1000)
    
    c_list_low.append(c)
    
    option_valuer = BinomOptionVal(S0,K,T,r,sigma,option_type='put',American=False,underlying='S',q=0)
    p = option_valuer.value(1000)

    p_list_low.append(p)

#Create the data frame
df_vol_smile = pd.DataFrame({'K':K_vec,
                                'call_high':c_list_high,
                                'call_low':c_list_low,
                                'put_high':p_list_high,
                                'put_low':p_list_low
                                })

# Calculate the final option value based on
# p = 0.4
df_vol_smile['call'] = df_vol_smile['call_high']*0.4 + \
    df_vol_smile['call_low']*0.6

df_vol_smile['put'] = df_vol_smile['put_high']*0.4 + \
    df_vol_smile['put_low']*0.6

# Use the final option values to compute an implied volatility

price_vec = df_vol_smile['call'].values.tolist()

S0 = 60
# Guess for sigma
sigma = 0.2

impl_vol_vec = []

for idx in range(len(price_vec)):
    option_valuer = BinomOptionVal(S0,K_vec[idx],T,r,sigma,option_type='call',American=False,underlying='S',q=0)
    impl_vol = option_valuer.impl_vol(price_vec[idx],500)

    impl_vol_vec.append(impl_vol)

df_vol_smile['impl_vol'] = impl_vol_vec

df_vol_smile['S0'] = 60

df_vol_smile['K/S0'] = df_vol_smile['K']/df_vol_smile['S0']


#%%
df_vol_smile

#%%

df_vol_smile.plot(x='K/S0',y='impl_vol',xlabel='K/S0',ylabel='Implied Volatility')


# %%
