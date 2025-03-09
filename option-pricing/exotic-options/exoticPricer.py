import numpy as np
from scipy.stats import norm
from copy import deepcopy

def barrier_pricer(S0: float,K: float,T: float,r: float,sigma: float,H: float,option_type='call',barrier_type='DI',q: float=0):
    
    # Firstly, define a function for a vanilla call/put
    def vanilla_price(S0,K,T,r,sigma,option_type,q):
        # Assign values for d1 and d2
        d1 = (np.log(S0/K)+(r+sigma**2/2)*T)/\
            (sigma*np.sqrt(T))
        
        d2 = d1 - sigma*np.sqrt(T)

        # Assign analytic option value based on type
        if option_type == 'call':
            value = S0*np.exp(-q*T)*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)    


        elif option_type == 'put':
            value = K*np.exp(-r*T)*norm.cdf(-d2)-S0*np.exp(-q*T)*norm.cdf(-d1)

        return value
    
    # Use if else logic to calculate the price depending
    # on H vs K -> Barrier Option type -> call or put 
    # on the barrier option type
    
    #First, define required constants
    lam = (r-q+sigma**2/2)/(sigma**2/2)
    
    y = np.log(H**2/(S0*K))/(sigma*np.sqrt(T))+lam*sigma*np.sqrt(T)
    
    x1 = np.log(S0/H)/(sigma*np.sqrt(T))+lam*sigma*np.sqrt(T)

    y1 = np.log(H/S0)/(sigma*np.sqrt(T))+lam*sigma*np.sqrt(T)

    # Condition for H <= K
    if H <= K:
            
        # down-and-in
        if barrier_type == 'DI':
            # call option
            if option_type == 'call':
                value = S0*np.exp(-q*T)*(H/S0)**(2*lam)*norm.cdf(y)-\
                    K*np.exp(-r*T)*(H/S0)**(2*lam-2)*norm.cdf(y-sigma*np.sqrt(T))
                return value

            # put option
            elif option_type == 'put':
                value = -S0*norm.cdf(-x1)*np.exp(-q*T)+\
                    K*np.exp(-r*T)*norm.cdf(-x1+sigma*np.sqrt(T))+\
                    S0*np.exp(-q*T)*(H/S0)**(2*lam)*(norm.cdf(y)-norm.cdf(y1))-\
                        -K*np.exp(-r*T)*(H/S0)**(2*lam-2)\
                            *(norm.cdf(y-sigma*np.sqrt(T))-norm.cdf(y1-sigma*np.sqrt(T)))                
                return value

        # down-and-out
        elif barrier_type == 'DO':
            # call option
            if option_type == 'call':
                c = vanilla_price(S0,K,T,r,sigma,option_type,q)
                
                value = S0*np.exp(-q*T)*(H/S0)**(2*lam)*norm.cdf(y)-\
                    K*np.exp(-r*T)*(H/S0)**(2*lam-2)*norm.cdf(y-sigma*np.sqrt(T))
                
                value = c - value
                
                return value
            
            # put option
            elif option_type == 'put':
                p = vanilla_price(S0,K,T,r,sigma,option_type,q)

                value = -S0*norm.cdf(-x1)*np.exp(-q*T)+\
                    K*np.exp(-r*T)*norm.cdf(-x1+sigma*np.sqrt(T))+\
                    S0*np.exp(-q*T)*(H/S0)**(2*lam)*(norm.cdf(y)-norm.cdf(y1))-\
                        -K*np.exp(-r*T)*(H/S0)**(2*lam-2)\
                            *(norm.cdf(y-sigma*np.sqrt(T))-norm.cdf(y1-sigma*np.sqrt(T)))
                
                value = p - value

                return value

        # up-and-in
        elif barrier_type == 'UI':
            # call option
            if option_type == 'call':
                c = vanilla_price(S0,K,T,r,sigma,option_type,q)
                return c

            # put option
            elif option_type == 'put':
                p = vanilla_price(S0,K,T,r,sigma,option_type,q)
                value = -S0*norm.cdf(-x1)*np.exp(-q*T)\
                    +K*np.exp(-r*T)*norm.cdf(-x1+sigma*np.sqrt(T))+\
                    S0*np.exp(-q*T)*(H/S0)**(2*lam)*norm.cdf(-y1)-K*np.exp(-r*T)*\
                    (H/S0)**(2*lam-2)*norm.cdf(-y1+sigma*np.sqrt(T))

                value = p - value
                return value                   
            
        # up-and-out
        elif barrier_type == 'UO':
            # call option
            if option_type == 'call':
                return 0.0

            # put option
            elif option_type == 'put':
                value = -S0*norm.cdf(-x1)*np.exp(-q*T)\
                    +K*np.exp(-r*T)*norm.cdf(-x1+sigma*np.sqrt(T))+\
                    S0*np.exp(-q*T)*(H/S0)**(2*lam)*norm.cdf(-y1)-K*np.exp(-r*T)*\
                    (H/S0)**(2*lam-2)*norm.cdf(-y1+sigma*np.sqrt(T))
                return value    
                
        
    # Condition for H > K
    else:
        # down-and-in
        if barrier_type == 'DI':
            # call option
            if option_type == 'call':
                c = vanilla_price(S0,K,T,r,sigma,option_type,q)

                value = S0*norm.cdf(x1)*np.exp(-q*T)\
                    -K*np.exp(-r*T)*norm.cdf(x1-sigma*np.sqrt(T))-\
                    S0*np.exp(-q*T)*(H/S0)**(2*lam)*norm.cdf(y1)+\
                    K*np.exp(-r*T)*(H/S0)**(2*lam-2)*norm.cdf(y1-sigma*np.sqrt(T))
                
                value = c - value
                
                return value
            # put option
            elif option_type == 'put':
                p = vanilla_price(S0,K,T,r,sigma,option_type,q)
                return p


        # down-and-out
        elif barrier_type == 'DO':
            # call option
            if option_type == 'call':
                value = S0*norm.cdf(x1)*np.exp(-q*T)\
                    -K*np.exp(-r*T)*norm.cdf(x1-sigma*np.sqrt(T))-\
                    S0*np.exp(-q*T)*(H/S0)**(2*lam)*norm.cdf(y1)+\
                    K*np.exp(-r*T)*(H/S0)**(2*lam-2)*norm.cdf(y1-sigma*np.sqrt(T))
                return value
            # put option
            elif option_type == 'put':
                return 0

        # up-and-in
        elif barrier_type == 'UI':
            # call option
            if option_type == 'call':
                value = S0*norm.cdf(x1)*np.exp(-q*T)-K*np.exp(x1-sigma*np.sqrt(T))-S0*np.exp(-q*T)*(H/S0)**(2*lam)*(norm.cdf(-y)-norm.cdf(-y1))+\
                    K*np.exp(-r*T)*(H/S0)**(2*lam-2)*(norm.cdf(-y+sigma*np.sqrt(T))-norm.cdf(-y1+sigma*np.sqrt(T)))                

                return value
            # put option
            elif option_type == 'put':
                value = -S0*np.exp(-q*T)*(H/S0)**(2*lam)*norm.cdf(-y)+\
                    K*np.exp(-r*T)*(H/S0)**(2*lam-2)*norm.cdf(-y+sigma*np.sqrt(T))
        
        # up-and-out
        elif barrier_type == 'UO':
            # call option
            if option_type == 'call':
                
                c = vanilla_price(S0,K,T,r,sigma,option_type,q)

                value = S0*norm.cdf(x1)*np.exp(-q*T)-K*np.exp(x1-sigma*np.sqrt(T))-S0*np.exp(-q*T)*(H/S0)**(2*lam)*(norm.cdf(-y)-norm.cdf(-y1))+\
                    K*np.exp(-r*T)*(H/S0)**(2*lam-2)*(norm.cdf(-y+sigma*np.sqrt(T))-norm.cdf(-y1+sigma*np.sqrt(T)))                

                value = c - value
                return value


            # put option
            elif option_type == 'put':
                p = vanilla_price(S0,K,T,r,sigma,option_type,q)                

                value = -S0*np.exp(-q*T)*(H/S0)**(2*lam)*norm.cdf(-y)+\
                    K*np.exp(-r*T)*(H/S0)**(2*lam-2)*norm.cdf(-y+sigma*np.sqrt(T))

                value = p - value

                return value
            

if __name__=='__main__':

    # inputs:
    S0 = 19
    K = 20
    T = 3/12
    r = 0.05
    sigma = 0.4 
    H = 18
    option_type = 'call'
    barrier_type = 'DO'
    q = 0.05 


    # function call:
    price = barrier_pricer(S0,K,T,r,sigma,H,option_type,barrier_type,q)

    print(price)