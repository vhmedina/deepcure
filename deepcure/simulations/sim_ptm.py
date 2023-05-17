import numpy as np
import pandas as pd
from deepcure.simulations.utils import vinv_F_exponential
from deepcure.simulations.utils import tf_ortho

# --------------- Simulate data from Promotion Time Cure Model --------------- #
def simDF_PTM(scenario, n, end_of_study, lam=1, seed=12345):
    """Generate data for PTM simulation
    
    Arguments:
    - scenario: 0: linear, 1: 1 covariate, 2: 3 covariates, 3: 10 covariates, 4: scenario 2 + linear predictor
    - n: number of subjects
    - lam: lambda for exponential distribution
    - end_of_study: end of study
    - seed: random seed

    Returns:
    - data: dataframe with simulated data

    Reference:
    - Xie, Y., & Yu, Z. (2021). Promotion time cure rate model with a neural network estimated nonparametric component. Statistics in Medicine, 40(15), 3516–3532. https://doi.org/10.1002/sim.8980

    """
    # ------------------------------- Scenario = 0 ------------------------------- #
    if scenario == 0: # linear predictor
        np.random.seed(seed) # reproducibility
        num_covs=3 # number of covariates
        x = np.random.uniform(size=(n, num_covs)) # random value for covariates
        eta= 1.2*x[:,0]-0.5*x[:,1]-1.2*x[:,2] # generate predictor
        theta = np.exp(eta) # theta from PTM
        N_i = np.random.poisson(lam = theta, size = n) # number of potential risks (poisson distribution)
        time = vinv_F_exponential(lamb = lam, n=n, N_i=N_i, seed=seed) # simulate time to event
        time[np.isinf(time)] = end_of_study # set time to end of study if time is infinite
        censor_time =  np.random.uniform(0.05*end_of_study, 2*end_of_study, size=n) # censor time randomly between 5% and 200% of end of study
        event_time = np.min(np.stack((time, censor_time, end_of_study * np.ones(n))).T, axis=1) # observed event time is the min of true time, censor and end of study
        event_ind = np.ones(n) # event indicator
        event_ind[N_i==0]=0 # cure cases are not subject to events
        event_ind[np.where((event_time<time) & (N_i>0))]=0 # censor cases
        # create dataframe
        data = pd.DataFrame(np.concatenate([x, event_ind.reshape(-1,1),event_time.reshape(-1,1),N_i.reshape(-1,1), eta.reshape(-1,1)], axis = 1),  columns=['x1', 'x2', 'x3', 'label_event', 'time_event', 'N_i','eta'])

    # ------------------------------- Scenario = 1 ------------------------------- #
    if scenario == 1:
        np.random.seed(seed) # reproducibility
        x = np.random.uniform(size = n) # random value for covariate
        eta = np.log(0.15)+0.1*(3.5*10**3*x**2*(1.-x)**8+2.2*10**4*x**8*(1.-x)**3) # generate predictor (constant not needed)
        theta = np.exp(eta) # theta from PTM
        N_i = np.random.poisson(lam = theta, size = n) # number of potential risks (poisson distribution)
        time = vinv_F_exponential(lamb = lam, n=n, N_i=N_i, seed=seed) # simulate time to event
        time[np.isinf(time)] = end_of_study # set time to end of study if time is infinite
        censor_time =  np.random.uniform(0.05*end_of_study, 2*end_of_study, size=n) # censor time randomly between 5% and 200% of end of study
        event_time = np.min(np.stack((time, censor_time, end_of_study * np.ones(n))).T, axis=1) # observed event time is the min of true time, censor and end of study
        event_ind = np.ones(n) # event indicator
        event_ind[N_i==0]=0 # cure cases are not subject to events
        event_ind[np.where((event_time<time) & (N_i>0))]=0 # censor cases
        # create dataframe
        data = pd.DataFrame(np.concatenate([x.reshape(-1,1), event_ind.reshape(-1,1),event_time.reshape(-1,1),N_i.reshape(-1,1), eta.reshape(-1,1)], axis = 1),  columns=['x', 'label_event', 'time_event', 'N_i','eta'])

    # ------------------------------- Scenario = 2 ------------------------------- #
    elif scenario == 2:
        np.random.seed(seed) # reproducibility
        num_covs=3 # number of covariates
        x = np.random.uniform(size=(n, num_covs)) # random value for covariates
        eta= -0.8*x[:,0]**2 + 4*x[:,1]**3 - 0.75*np.cos(x[:,2]) # generate predictor
        theta = np.exp(eta) # theta from PTM
        N_i = np.random.poisson(lam = theta, size = n) # number of potential risks (poisson distribution)
        time = vinv_F_exponential(lamb = lam, n=n, N_i=N_i, seed=seed) # simulate time to event
        time[np.isinf(time)] = end_of_study # set time to end of study if time is infinite
        censor_time =  np.random.uniform(0.05*end_of_study, 2*end_of_study, size=n) # censor time randomly between 5% and 200% of end of study
        event_time = np.min(np.stack((time, censor_time, end_of_study * np.ones(n))).T, axis=1) # observed event time is the min of true time, censor and end of study
        event_ind = np.ones(n) # event indicator
        event_ind[N_i==0]=0 # cure cases are not subject to events
        event_ind[np.where((event_time<time) & (N_i>0))]=0 # censor cases
        # create dataframe
        data = pd.DataFrame(np.concatenate([x, event_ind.reshape(-1,1),event_time.reshape(-1,1),N_i.reshape(-1,1), eta.reshape(-1,1)], axis = 1),  columns=['x1', 'x2', 'x3', 'label_event', 'time_event', 'N_i','eta'])

    # ------------------------------- Scenario = 3 ------------------------------- #
    elif scenario == 3:
        np.random.seed(seed) # reproducibility
        num_covs=10 # number of covariates
        Sigma = [[1,0.8,0.5,0.2,0],[0.8,1,0.2,0.6,0],[0.5,0.2,1,0.3,0],[0.2,0.6,0.3,1,0],[0,0,0,0,1]] # covariance matrix
        mean = np.zeros(int(num_covs/2)) # mean vector
        x1 = np.random.multivariate_normal(mean,Sigma, n) # random value for covariates x1
        x2 = np.random.standard_normal(size=(n, int(num_covs/2))) # random value for covariates x2
        pred1 = 0.4*(0.05*x1[:,0]**2+0.05*np.tanh(x1[:,1])-0.05*x1[:,2]*x1[:,3]*(4-0.0005*x1[:,2]*x1[:,3])**2+np.log(np.abs(x1[:,0]+x1[:,4]))) # generate predictor1
        pred2=0.4*(0.05*x2[:,0]**2+0.05*np.tanh(x2[:,1])-0.05*x2[:,2]*x2[:,3]*(4-0.0005*x2[:,2]*x2[:,3])**2+np.log(np.abs(x2[:,0]+x2[:,4]))) # generate predictor2
        eta = pred1+pred2 # total predictor
        theta = np.exp(eta) # theta from PTM
        N_i = np.random.poisson(lam = theta, size = n) # number of potential risks (poisson distribution)
        time = vinv_F_exponential(lamb = lam, n=n, N_i=N_i, seed=seed) # simulate time to event
        time[np.isinf(time)] = end_of_study # set time to end of study if time is infinite
        censor_time =  np.random.uniform(0.05*end_of_study, 2*end_of_study, size=n) # censor time randomly between 5% and 200% of end of study
        event_time = np.min(np.stack((time, censor_time, end_of_study * np.ones(n))).T, axis=1) # observed event time is the min of true time, censor and end of study
        event_ind = np.ones(n) # event indicator
        event_ind[N_i==0]=0 # cure cases are not subject to events
        event_ind[np.where((event_time<time) & (N_i>0))]=0 # censor cases
        # create dataframe
        data = pd.DataFrame(np.concatenate([x1, x2,event_ind.reshape(-1,1),event_time.reshape(-1,1),N_i.reshape(-1,1),eta.reshape(-1,1)], axis = 1),  columns=['x11', 'x12', 'x13','x14','x15','x21','x22','x23','x24','x25', 'label_event', 'time_event', 'N_i','eta'])

    # -------------------------------- Scenario 4 -------------------------------- #
    elif scenario == 4:
        np.random.seed(seed) # reproducibility
        num_covs=3 # number of covariates
        integ = - 1
        betas = np.array([2 / (i + 1) for i in range(num_covs)])
        x = np.random.uniform(-1,1,size=(n, num_covs)) # random value for covariates
        ## generate linear predictor part
        lin_eta = integ + np.dot(x, betas)
        ## generate non-linear predictor part
        xbase=np.hstack([x[:,0:1]**2, x[:,1:2]**3, np.cos(x[:,2:3])])
        x_w_int = np.concatenate((np.ones((n, 1)), x), axis=1)
        xbase = tf_ortho(xbase,x_w_int)
        non_lin_eta = np.dot(xbase, np.array([-0.8, 4, -0.75]))
        # non_lin_eta = non_lin_eta - np.mean(non_lin_eta)
        ## generate total predictor
        eta = lin_eta + non_lin_eta # generate predictor
        theta = np.exp(eta) # theta from PTM
        N_i = np.random.poisson(lam = theta, size = n) # number of potential risks (poisson distribution)
        time = vinv_F_exponential(lamb = lam, n=n, N_i=N_i, seed=seed) # simulate time to event
        time[np.isinf(time)] = end_of_study # set time to end of study if time is infinite
        censor_time =  np.random.uniform(0.05*end_of_study, 2*end_of_study, size=n) # censor time randomly between 5% and 200% of end of study
        event_time = np.min(np.stack((time, censor_time, end_of_study * np.ones(n))).T, axis=1) # observed event time is the min of true time, censor and end of study
        event_ind = np.ones(n) # event indicator
        event_ind[N_i==0]=0 # cure cases are not subject to events
        event_ind[np.where((event_time<time) & (N_i>0))]=0 # censor cases
        # create dataframe
        data = pd.DataFrame(np.concatenate([x, event_ind.reshape(-1,1),event_time.reshape(-1,1),N_i.reshape(-1,1), eta.reshape(-1,1)], axis = 1),  columns=['x1', 'x2', 'x3', 'label_event', 'time_event', 'N_i','eta'])
   
    # ----------------------------- Print some stats ----------------------------- #
    print(f"Number of subjects: {n}")
    print(f"Number of events: {int(np.sum(event_ind))}")
    print(f"Event rate (#events/n): {np.sum(event_ind)/n}")

    return data

