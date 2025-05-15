import numpy as np
from scipy.stats import norm
from copy import deepcopy

class BinomOptionVal:
    '''A class representing an option of type put/call and American/European, 
    used for valuing the option using the Binomial Pricing Method. Takes inputs of 
    underlying stock price (S0), strike price (K), time to maturity in years (T),
    risk-free-rate (r), underlying stock volatility (sigma), whether the option is a put or a call
    (option_type) and a boolean which represents an American option if True, European if False (American).
    The underlying asset can be set as 'S' for stock/equity or 'F' for futures.
    for dividend paying stock (equivalent to foreign exchange and stock indices) (underlying).  
    Method 'value' can be used to perform the valuation of the option,
    returning string output of the option value and key information.'''
    
    def __init__(self,S0: float,K: float,T: float,r: float,sigma: float,option_type='call',American=False,underlying='S',q: float=0):
        
        ################
        #Error Handling#
        ################
        if option_type not in ['call','put']:
            raise ValueError("Option type must be either 'put' or 'call'")
        
        if underlying not in ['S','F']:
            raise ValueError("Underlying asset must be either 'S' or 'F'")
        
        if S0<=0 or K<=0 or T <= 0 or r <= 0 or sigma <= 0:
            raise ValueError("Input values must be strictly positive.")

        ############
        #Attributes#
        ############
        # Define the attributes from inputs
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type
        self.American = American
        self.underlying = underlying
        self.q = q

    def value_bsm(self,div: list[float]=[],T_div: list[float]=[])->float:
        '''Values put and call options using the Black-Scholes-Merton model.
        Can only be applied to European options. An optional list
        of dividend amounts (div) and their time to ex dividend dates 
        (T_div) can be provided'''

        if (self.option_type=='put' and self.American) or self.underlying != 'S':
            raise ValueError("Cannot apply BSM to American puts or underlying other than stock.")


        sum_divs = 0 # initialise sum diviend value
        if len(div)>0 or len(T_div)>0:
            if len(div) != len(T_div):
                raise ValueError("Must be the same number of dividend payments and times.")
            
            # Obtain the sum value of the dividends
            for i in range(len(div)):
                sum_divs += div[i]*np.exp(-self.r*T_div[i])


        # Calculate the value of d1 and d2
        d1 = (np.log((self.S0-sum_divs)/self.K)+(self.r-self.q+self.sigma**2/2)*self.T)/(self.sigma*np.sqrt(self.T))

        d2 = d1 - self.sigma*np.sqrt(self.T)

        # Set up type for printing
        type = 'European'

        #Perform valuation based on type
        if self.option_type == 'call':
            value =  (self.S0*np.exp(-self.q*self.T)-sum_divs)*norm.cdf(d1)-self.K*np.exp(-self.r*self.T)*norm.cdf(d2)
            # Incorporate Black's approximation if American call

            if self.American:
                #redefine type
                type = 'American'
                # Compute the value if exercised at the final ex-dividend date
                # Recompute d1 and d2
                d1 = (np.log((self.S0-sum_divs+div[-1]*np.exp(-self.r*T_div[-1]))/self.K)+(self.r-self.q+self.sigma**2/2)*T_div[-1])/(self.sigma*np.sqrt(T_div[-1]))
                d2 = d1 - self.sigma*np.sqrt(T_div[-1])
                # Compute valuation of early exercise
                value_early = (self.S0*np.exp(-self.q*self.T)-sum_divs+div[-1]*np.exp(-self.r*T_div[-1]))*norm.cdf(d1)-self.K*np.exp(-self.r*T_div[-1])*norm.cdf(d2)
                # Redefine value as maximum of the Euro value and American value
                value = max(value,value_early)
            # Round the value to 4 d.p.
            value = round(10000*value)/10000

        elif self.option_type == 'put':
            value = self.K*np.exp(-self.r*self.T)*norm.cdf(-d2)-(self.S0*np.exp(-self.q*self.T)-sum_divs)*norm.cdf(-d1)
            value = round(1000000*value)/1000000
        # Print the key information
        print(f'Option value of type {type} {self.option_type} and maturity of T={self.T} years\n is ${value}')
        
        #############################
        # Assign values of the Greeks

        # Prepare common call-put values
        # Theta intermediates
        d1dT = -np.log((self.S0-sum_divs)/self.K)*0.5/self.sigma*(self.T)**(-3/2) +\
            0.5*(self.r-self.q+self.sigma**2/2)/(self.sigma*self.T**0.5)

        d2dT = -np.log((self.S0-sum_divs)/self.K)*0.5/self.sigma*(self.T)**(-3/2) +\
            0.5*(self.r-self.q-self.sigma**2/2)/(self.sigma*self.T**0.5)

        # Vega intermediates
        d1dsig = -(np.log((self.S0-sum_divs)/self.K)/(self.sigma**2*np.sqrt(self.T)))-((self.r-self.q)*np.sqrt(self.T)/(self.sigma**2))+np.sqrt(self.T)/2
        d2dsig = d1dsig-np.sqrt(self.T)  

        if self.option_type=='call':
            # Delta
            self.delta = np.exp(-self.q*self.T)*norm.cdf(d1)    
            # Gamma
            self.gamma = np.exp(-self.q*self.T)/(self.sigma*self.S0*np.sqrt(self.T*2*np.pi))*np.exp(-d1**2/2)
            # Theta
            self.theta = -(self.S0*np.exp(-self.q*self.T)*(d1dT/np.sqrt(2*np.pi)*np.exp(-d1**2/2)-self.q*norm.cdf(d1))+\
                -self.K*np.exp(-self.r*self.T)*(d2dT/np.sqrt(2*np.pi)*np.exp(-d2**2/2)-self.r*norm.cdf(d2)))
            # Vega
            self.vega = d1dsig*((self.S0-sum_divs)*np.exp(-self.q*self.T)/np.sqrt(2*np.pi))*np.exp(-0.5*d1**2)\
                -d2dsig*(self.K*np.exp(-self.r*self.T)/np.sqrt(2*np.pi))*np.exp(-0.5*d2**2)
            # Rho
            self.rho = (self.S0-sum_divs)*np.exp(-self.q*self.T)*np.sqrt(self.T)/self.sigma/np.sqrt(2*np.pi)*np.exp(-d1**2/2)\
                +self.T*self.K*np.exp(-self.r*self.T)*norm.cdf(d2)+\
                -self.K*np.exp(-self.r*self.T)*(np.sqrt(self.T)/self.sigma)/np.sqrt(2*np.pi)*np.exp(-d2**2/2)

        elif self.option_type=='put':
            # Delta
            self.delta = np.exp(-self.q*self.T)*(norm.cdf(d1)-1)    
            # Gamma
            self.gamma = np.exp(-self.q*self.T)/(self.sigma*self.S0*np.sqrt(self.T*2*np.pi))*np.exp(-d1**2/2)
            # Theta
            self.theta = -(self.S0*np.exp(-self.q*self.T)*(d1dT/np.sqrt(2*np.pi)*np.exp(-d1**2/2)+self.q*norm.cdf(-d1))+\
                -self.K*np.exp(-self.r*self.T)*(d2dT/np.sqrt(2*np.pi)*np.exp(-d2**2/2)+self.r*norm.cdf(-d2)))
            # Vega
            self.vega = -d1dsig*((self.S0-sum_divs)*np.exp(-self.q*self.T)/np.sqrt(2*np.pi))*np.exp(-0.5*d1**2)\
                +d2dsig*(self.K*np.exp(-self.r*self.T)/np.sqrt(2*np.pi))*np.exp(-0.5*d2**2)
            # Rho
            self.rho = (self.S0-sum_divs)*np.exp(-self.q*self.T)*np.sqrt(self.T)/self.sigma/np.sqrt(2*np.pi)*np.exp(-d1**2/2)\
                -self.T*self.K*np.exp(-self.r*self.T)*norm.cdf(-d2)+\
                -self.K*np.exp(-self.r*self.T)*(np.sqrt(self.T)/self.sigma)/np.sqrt(2*np.pi)*np.exp(-d2**2/2)
        #############################
        
        # Return the option value
        return value
    
    def value(self,N: int,div: list[float]=[],T_div: list[float]=[],div_yield: bool=False)->float:
        '''A method to perform the valuation of the option using
        the binomial pricing method. Takes number of time step nodes (N) as an input.
        Creates attributes of the stock price tree (stockTree) and option value tree(valTree)
        and prints key option information after completing valuation.'''
        
        ################
        #Error Handling#
        ################
        if N <=1:
            raise ValueError("Number of time step nodes must be strictly greater than 1")
        
        
        #####################################
        #Variable Assignment and Calculation#
        #####################################

        # Assign number of nodes
        self.N = N

        # Calculate parameters
        self.dt = self.T/(N-1) # time step

        if len(div)>0 or len(T_div)>0:
            if len(div) != len(T_div):
                raise ValueError("Must be the same number of dividend payments and times.")
            
            # Find the node at which to subtract the dividend for each T_div
            div_node = []
            for T_val in T_div:
                div_node.append(int((T_val/self.dt)+1))


        self.u = np.exp(self.sigma*np.sqrt(self.dt)) #upward movement proportion

        self.d = 1/self.u #downward movement proportion

        #Discount Factor
        self.DF = np.exp(-self.r*self.dt)

        if self.underlying =='S':
            #probability upward movement
            self.p = (np.exp((self.r)*self.dt)-self.d)/(self.u-self.d) #probability of upward movement

        elif self.underlying=='F':
            #probability upward movement
            self.p = (1-self.d)/(self.u-self.d) #probability of upward movement
        
        #Tree initialisation
        self.stockTree = np.zeros((self.N,self.N))
        self.valTree = np.zeros((self.N,self.N))
        
        ##################
        #Stock Tree Calcs#
        ##################

        # Set the stock values at each time step.
        # Index i represents the time step.
        # Index j represents the number of upward movements 
        # in the underlying stock.

        # Iterate through time steps from 
        # start to finish
        # 
        # Also initialise vector for dividend values containing N nodes.
        divVector = np.zeros((self.N,))
        
        # Determine if early exercise for either ex-dividend date
        if len(div)>0 or len(T_div)>0:
           # Create counter for the dividend values,
           # Increment up for each dividend
            counter = 0
            for node in div_node:
                # Assign dividend to the vector
                divVector[node] = div[counter]
                # increment the counter
                counter+=1 

        # If the dividend is in yield form
        # create vector of 1-yield (i.e. (1-delta_i))
        # 
        # Should remain as 1 if all div yields are 0
        if div_yield and len(div)>0:
            divYieldVec = np.cumprod(np.ones((self.N,))-divVector)
        else:
            divYieldVec = np.ones((self.N,))
        
        # Stock tree population
        for i in range(self.N):
            # Create discounted dividend amount
            if len(div)>0 and not div_yield:
                # Compute discounted dividend amounts
                # Initialise index and dividend amount sum at 0
                idx = 0
                div_sum = 0
                # Add together the discounted dividend amounts
                for div_amt in div:
                    if T_div[idx] > i*self.dt: 
                        div_sum += div_amt*np.exp(-self.r*(T_div[idx]-i*self.dt))
                    idx += 1
            else:
                div_sum = 0
            
            # Iterate through the tree
            if i==0:
                self.stockTree[i][i]=self.S0 - div_sum # Subtract dividends for first entry 
            else:
                # For other nodes, there can be a max of i upward
                # movements per node, a min of 0 
                for j in range(i+1):
                    # For LHS stock prices, same number of upward movement
                    # means downward movement at last node
                    self.stockTree[i][j] = self.stockTree[0][0]*self.u**j*self.d**(i-j)*divYieldVec[i]+div_sum
        
        #Return to nominal value
        self.stockTree[0][0] = self.S0
        ##################
        #Value Tree Calcs#
        ##################
        
        # Get the value if option is exercised based on the final Stock values
        if self.option_type=='call':
            self.valTree[-1,:] = np.maximum(self.stockTree[-1,:]-self.K,0)
        elif self.option_type=='put':
            self.valTree[-1,:] = np.maximum(self.K-self.stockTree[-1,:],0)

        # Assign option value at each node, 
        # working backwards through the tree,
        # Start at N-1 since final node 
        # has already been calculated above.
        
        for idx in range(1,self.N):
            # Represents time-step nodes in reverse order
            i = (self.N-1)-idx
            # Calculate values for each potential number of 
            # up movements at the node
            
            for j in range(i+1):
                # For American Options, take the max of the option value
                # and pay off from early exercise, otherwise just take the value
                if self.American and self.option_type=='put':
                    self.valTree[i,j] = max(self.DF*(self.p*self.valTree[i+1,j+1] + (1-self.p)*self.valTree[i+1,j]),self.K-self.stockTree[i,j])
                elif self.American and self.option_type=='call':          
                    self.valTree[i,j] = max(self.DF*(self.p*self.valTree[i+1,j+1] + (1-self.p)*self.valTree[i+1,j]),self.stockTree[i,j]-self.K)
                else:
                    self.valTree[i,j] = self.DF*(self.p*self.valTree[i+1,j+1] + (1-self.p)*self.valTree[i+1,j])
        
        #####################
        # Obtain Option Value#
        #####################
        self.optionValue = round(1000000*self.valTree[0][0])/1000000

        # Return the value of the option
        return self.optionValue

    def getTreeGreeks(self,N: int,div: list[float]=[],T_div: list[float]=[],div_yield: bool=False)->None:
        '''Calculates the option Greeks using discrete values as they appear 
        from the binomial tree method. Takes number of time step nodes (N) as an input.
        Assigns the objects Greek values as per the discrete tree values. 
        '''
        #######################
        # Obtain Option Greeks#
        #######################
        
        # Rho and Vega #
        ################
        # Perturbation constant
        d_r_sig = 1E-6
        # Perturbation for Vega #
        #########################
        # Obtain the option value with perturbed sigma 
        # perturb sigma
        self.sigma += d_r_sig
        # Compute option value perturbing sigma 
        f1_vega = self.value(N,div,T_div,div_yield)
        # Return to original value
        self.sigma -= d_r_sig

        # Perturbation for Rho #
        #########################
        # Obtain the option value with perturbed r 
        # perturb r
        self.r += d_r_sig
        # Compute option value perturbing sigma 
        f1_rho = self.value(N,div,T_div)
        # Return to original value
        self.r -= d_r_sig

        # Compute nominal value of the option
        f0 = self.value(N,div,T_div)

        #Vega#
        ######
        self.vega = (f1_vega-f0)/d_r_sig
        
        #Rho#
        #####
        self.rho = (f1_rho-f0)/d_r_sig

        # Delta #
        #########
        self.delta = (self.valTree[1,1] - self.valTree[1,0])/(self.S0*(self.u-self.d))

        # Gamma #
        #########
        self.gamma = ((self.valTree[2,2]-self.valTree[2,1])/(self.S0*self.u**2-self.S0)-\
            (self.valTree[2,1]-self.valTree[2,0])/(self.S0-self.S0*self.d**2))/(0.5*(self.S0*self.u**2-self.S0*self.d**2))

        # Theta #
        #########
        self.theta = (self.valTree[2,1]-self.valTree[0,0])/(2*self.dt)
        # Output option value and greeks
        print(f'Option Value is {self.optionValue}\nOption Greeks are:\nDelta: {self.delta}\nGamma: {self.gamma}\nTheta: {self.theta}\nVega: {self.vega}\nRho: {self.rho}')

        return None


    def impl_vol(self,price: float,N: int,div: list[float]=[],T_div: list[float]=[],div_yield: bool=False)->float:
        '''Given an option price, determines the implied volatility using initial sigma as an initial guess.
        Applies bisection method upon the binomial tree valuation method (with input N as number of nodes)
        to find implied volatility.
        '''

        sigma_guess = deepcopy(self.sigma)
        value_iter = self.value(N,div,T_div,div_yield)
        # Intial low guess
        sigma_low = 0.01
        # Initial high guess
        sigma_high = 0.99
        # Initial Error Calculation
        err = np.abs(value_iter-price)
        
        while err > 1E-4:
            if value_iter > price:
                if self.option_type=='call':
                    sigma_high = deepcopy(sigma_guess)
                elif self.option_type=='put':
                    sigma_high = deepcopy(sigma_guess)
            elif value_iter < price:
                if self.option_type=='call':
                    sigma_low = deepcopy(sigma_guess)
                elif self.option_type=='put':
                    sigma_low = deepcopy(sigma_guess)

            sigma_guess = (sigma_low+sigma_high)/2
            self.sigma = deepcopy(sigma_guess)
            value_iter = self.value(N,div,T_div,div_yield)
            err = np.abs(value_iter-price)
            print(value_iter)
        
        print(f'Implied volatility is {self.sigma}')
        return self.sigma

    
    def impl_vol_bsm(self,price: float,div: list[float]=[],T_div: list[float]=[])->float:
        '''Using the provided value of volatility as an initial guess,
        and taking the input of option price, this method uses the Black-Scholes-Merton model
        and  Newton-Rhapson method to find 
        the implied volatility of a European call or put option with or without dividends. Currently only works 
        with European options.'''
        
        ################
        #Error Checking#
        ################

        # Ensure it is a European Option
        if self.American:
            raise ValueError('Can only be performed on European call or put options')
        
        # If using dividends, check that each ex-dividend date has a corresponding
        # dividend payment amount
        if len(div)>0 or len(T_div)>0:
            if len(div) != len(T_div):
                raise ValueError("Must be the same number of dividend payments and times.")
        
        #Dividend Adjustment
        sum_divs = 0 # initialise sum diviend value
        if len(div)>0 or len(T_div)>0:
            if len(div) != len(T_div):
                raise ValueError("Must be the same number of dividend payments and times.")
            
            # Obtain the sum value of the dividends
            for i in range(len(div)):
                sum_divs += div[i]*np.exp(-self.r*T_div[i])


        # Create lambda functions for the value d1 and d2
        d1 = lambda sigma_var: (np.log((self.S0-sum_divs)/self.K)+(self.r-self.q+sigma_var**2/2)*self.T)/(sigma_var*np.sqrt(self.T))

        d2 = lambda sigma_var: d1(sigma_var) - sigma_var*np.sqrt(self.T)

        # Obtain functions and derivative for call option 
        # First, get derivatives for d1 and d2
        dd1 = lambda sigma_var: -(np.log((self.S0-sum_divs)/self.K)/(sigma_var**2*np.sqrt(self.T)))-((self.r-self.q)*np.sqrt(self.T)/(sigma_var**2))+np.sqrt(self.T)/2
        dd2 = lambda sigma_var: dd1(sigma_var)-np.sqrt(self.T) 

        #Now, create the value functions depending on if put or call
        if self.option_type=='call':
            # Option value function
            v = lambda sigma_var: (self.S0*np.exp(-self.q*self.T)-sum_divs)*norm.cdf(d1(sigma_var))-self.K*np.exp(-self.r*self.T)*norm.cdf(d2(sigma_var))-price

            # Option Vega
            dv = lambda sigma_var: dd1(sigma_var)*((self.S0*np.exp(-self.q*self.T)-sum_divs)/np.sqrt(2*np.pi))*np.exp(-0.5*d1(sigma_var)**2)\
                -dd2(sigma_var)*(self.K*np.exp(-self.r*self.T)/np.sqrt(2*np.pi))**np.exp(-0.5*d2(sigma_var)**2)
            
        elif self.option_type=='put':
            # Option value function
            v = lambda sigma_var: self.K*np.exp(-self.r*self.T)*norm.cdf(-d2(sigma_var))-(self.S0*np.exp(-self.q*self.T)-sum_divs)*norm.cdf(-d1(sigma_var))-price

            # Option Vega
            dv = lambda sigma_var: -dd2(sigma_var)*(self.K*np.exp(-self.r*self.T)/np.sqrt(2*np.pi))**np.exp(-0.5*d2(sigma_var)**2)\
                +dd1(sigma_var)*((self.S0*np.exp(-self.q*self.T)-sum_divs)/np.sqrt(2*np.pi))*np.exp(-0.5*d1(sigma_var)**2)

        # Newton-Rhapson iterator:
        def NR_iter(sigma_iter):
            return sigma_iter-v(sigma_iter)/dv(sigma_iter)
        
        imp_vol = NR_iter(self.sigma)
        err = np.abs(v(imp_vol))
        
        #Iterate until converge to solution
        while err > 1E-6:
            imp_vol = NR_iter(imp_vol)
            err = np.abs(v(imp_vol))
            

        print(f'Option value of type European {self.option_type} and maturity of T={self.T} years\n and price ${price} has implied volatility of {imp_vol}')
    
        return imp_vol
    
if __name__=='__main__':
    
    S0 = 334.07
    r = 0.0452
    sigma =0.25
    K = 315
    T = 1

    optionPricer = BinomOptionVal(S0,K,T,r,sigma,option_type='put',American=False,underlying='S')
    
    imp_vol = optionPricer.impl_vol(20,N=1000)

    print(imp_vol)

