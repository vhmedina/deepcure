import numpy as np
from numba import prange, njit
from deepcure.performance.utils import _get_stats

@njit
def get_auc(prob_cured, nsteps=100):
    """ Calculate the AUC using the survival cured estimate, based on Xie and Yu (2021).

    Args:
    - prob_cured: Probability of being cured.
    - nsteps: Number of steps to use in the trapezoidal rule.

    Returns:
    - scalar: AUC.

    References:
    Xie, Y., & Yu, Z. (2021). Promotion time cure rate model with a neural network estimated nonparametric component. Statistics in Medicine, 40(15), 3516–3532. https://doi.org/10.1002/sim.8980

    """
    TPR = np.array(())
    FPR = np.array(())

    for c in np.linspace(0,1,nsteps):
        tpr = np.sum(np.where(prob_cured<=c,1,0)*(1-prob_cured))/np.sum((1-prob_cured))
        fpr = np.sum(np.where(prob_cured<=c,1,0)*(prob_cured))/np.sum((prob_cured))
        TPR = np.append(TPR, tpr)
        FPR = np.append(FPR, fpr)
    return np.sum((TPR[:(nsteps-1)]+ TPR[1:])*(FPR[1:]-FPR[:(nsteps-1)])/2)


# Define function to calculate concordance index
@njit(parallel=True)
def get_concordance_index(surv_mat, time_event, label_event):
    """ Calculate the concordance index based on Antolini et al.

    Args:
    - surv_mat: Matrix with the probability of survival (n_ind, ti_ind).
    - time_event: Time of event (or censoring).
    - label_event: Label of event (0 if censored).

    Returns:
    - scalar Ctd-index
    
    References:
    Antolini, L., Boracchi, P., & Biganzoli, E. (2005). A time-dependent discrimination index for survival data. Statistics in Medicine, 24(24), 3927–3944. https://doi.org/10.1002/sim.2427

    """
    # reshape
    time_event = time_event.reshape(-1)
    label_event = label_event.reshape(-1)
    # estimate the comparable pairs
    n_ind = surv_mat.shape[0]
    pi_com = 0
    pi_conc = 0
    for i in prange(n_ind):
        for j in prange(n_ind):
            if j!=i:
                # condition to be comparable
                if (time_event[i]<time_event[j] and label_event[i]==1) or (time_event[i]==time_event[j] and label_event[i]==1 and label_event[j]==0):
                    pi_com += 1
                    # condition to be concordant
                    if surv_mat[i,i]<surv_mat[j,i]:
                        pi_conc += 1
    return pi_conc/pi_com

# concordance index ind
@njit(parallel=True)
def get_concordance_index_ind(surv_mat_i, index_i, time_event, label_event):
    """ Calculate the concordance index based on Antolini et al.

    Args:
    - surv_mat_i: Matrix with the probability of survival (n_ind, 1).
    - index_i: Index of the individual.
    - time_event: Time of event (or censoring).
    - label_event: Label of event (0 if censored).

    Returns:
    - scalar Ctd-index for the individual i
    
    References:
    Antolini, L., Boracchi, P., & Biganzoli, E. (2005). A time-dependent discrimination index for survival data. Statistics in Medicine, 24(24), 3927–3944. https://doi.org/10.1002/sim.2427

    """
    # reshape
    time_event = time_event.reshape(-1)
    label_event = label_event.reshape(-1)
    # estimate the comparable pairs
    n_ind = time_event.shape[0]
    pi_com = 0
    pi_conc = 0
    for j in prange(n_ind):
        if j!=index_i:
            # condition to be comparable
            if (time_event[index_i]<time_event[j] and label_event[index_i]==1) or (time_event[index_i]==time_event[j] and label_event[index_i]==1 and label_event[j]==0):
                pi_com += 1
                # condition to be concordant
                if surv_mat_i[index_i] < surv_mat_i[j]:
                    pi_conc += 1
    return pi_conc,pi_com

# Define condition to be comparable
@njit
def _cond_comparable(t1,t2,l1,l2):
    """ Condition to be comparable.

    Args:
    - t1: Time of event (or censoring) for subject 1.
    - t2: Time of event (or censoring) for subject 2.
    - l1: Label of event (0 if censored) for subject 1.
    - l2: Label of event (0 if censored) for subject 2.

    Returns:
    - boolean: True if comparable, False otherwise.

    """
    return (((t1<t2) & (l1==1)) | ((t1==t2) & (l1==1) & (l2==0)))

# Define condition to be concordant
@njit
def _cond_concordant(s1,s2):
    """ Condition to be concordant, given that are comparable.

    Args:
    - s1: Probability of survival for subject 1.
    - s2: Probability of survival for subject 2.

    Returns:
    - boolean: True if concordant, False otherwise.

    """
    return s1<s2

# concordance index ind ord
@njit(parallel=True)
def get_concordance_index_ind_ord(surv_mat_i_ord, i_ord, i_aux, time_event, label_event, time_ind_ord):
    """ Calculate the concordance index based on Antolini et al.

    Args:
    - surv_mat_i_ord: Matrix with the probability of survival (n_ind, 1).
    - i_ord: Index of the individual.
    - i_aux: Index of the individual in the original order.
    - time_event: Time of event (or censoring).
    - label_event: Label of event (0 if censored).
    - time_ind_ord: Index of the individual ordered by time.

    Returns:
    - scalar Ctd-index for the individual i
    
    References:
    Antolini, L., Boracchi, P., & Biganzoli, E. (2005). A time-dependent discrimination index for survival data. Statistics in Medicine, 24(24), 3927–3944. https://doi.org/10.1002/sim.2427

    """
    # reshape if needed
    if time_event.ndim>1:
        time_event = time_event.reshape(-1)
    if label_event.ndim>1:
        label_event = label_event.reshape(-1)
    # estimate the comparable pairs
    n_ind = time_event.shape[0]
    pi_com = 0
    pi_conc = 0
    # for j in range(i_aux+1,n_ind):
    for j in prange((n_ind-i_aux-1)):
        j_aux = i_aux+j+1
        # condition to be comparable
        if _cond_comparable(time_event[i_ord],time_event[time_ind_ord[j_aux]],label_event[i_ord],label_event[time_ind_ord[j_aux]]):
            pi_com += 1
            # condition to be concordant
            if _cond_concordant(surv_mat_i_ord[i_ord],surv_mat_i_ord[time_ind_ord[j_aux]]):
                pi_conc += 1
    return pi_conc,pi_com

@njit
def _get_kaplan_meier(time_event, label_event, censor_dist=False):
    # get stats
    times, events, censors, at_risk = _get_stats(time_event,label_event)
    # calculate kaplan-meier
    if censor_dist:
        at_risk -= events
        events = censors
    kaplan_meier = np.cumprod((1-events/at_risk))
    # add 0,1
    times = np.append(-np.infty, times)
    # kaplan_meier = np.r_[1, kaplan_meier]
    kaplan_meier = np.append(1, kaplan_meier)
    return times, kaplan_meier

@njit
def _get_km_prob(time_event, label_event, times_eval, censor_dist=False):    
    times, prob = _get_kaplan_meier(time_event,label_event, censor_dist=censor_dist)
    fwd_ind = np.searchsorted(times, times_eval)
    equal = np.absolute(times[fwd_ind] - times_eval) < 1e-8
    fwd_ind[~equal] -= 1
    return prob[fwd_ind]

@njit
def get_brier_score(time_event, label_event, time_event_test, label_event_test, surv_mat, times_eval):
    """ Calculate the Brier score based on Graf et al. (1999).

    Args:
    - time_event: numpy array of shape (n_ind,)
    - label_event: numpy array of shape (n_ind,)
    - time_event_test: numpy array of shape (n_ind_test,)
    - label_event_test: numpy array of shape (n_ind_test,)
    - surv_mat: numpy array of shape (n_ind_test, n_time_eval)
    - times_eval: numpy array of shape (n_time_eval,)

    Returns:
    - eval_time, Brier score for each evaluation time.
    
    References:
    Graf, E., Schmoor, C., Sauerbrei, W., & Schumacher, M. (1999). Assessment and comparison of prognostic classification schemes for survival data. Statistics in Medicine, 18(17–18), 2529–2545. https://doi.org/10.1002/(SICI)1097-0258(19990915/30)18:17/18<2529::AID-SIM274>3.0.CO;2-5

    """
    # reshape if needed
    if time_event.ndim>1:
        time_event = time_event.reshape(-1)
    if label_event.ndim>1:
        label_event = label_event.reshape(-1)
    if time_event_test.ndim>1:
        time_event_test = time_event_test.reshape(-1)
    if label_event_test.ndim>1:
        label_event_test = label_event_test.reshape(-1)
    # Get censored survival probability
    p_cens_train_ = _get_km_prob(time_event, label_event, times_eval, censor_dist=True)
    p_cens_train_[p_cens_train_ == 0] = np.inf
    p_cens_test_ = _get_km_prob(time_event, label_event, time_event_test, censor_dist=True)
    p_cens_test_[p_cens_test_ == 0] = np.inf
    
    brier = np.empty(times_eval.shape[0])
    for t_ind, t in enumerate(times_eval):
        surv_prob_t = surv_mat[:, t_ind]
        dead_t = ((time_event_test <= t) & (label_event_test==1))
        not_dead_t = (time_event_test > t)
        dead_cont = np.square(surv_prob_t) * dead_t / p_cens_test_
        not_dead_cont = np.square(1.0 - surv_prob_t) * not_dead_t / p_cens_train_[t_ind]
        brier[t_ind] = np.mean(dead_cont + not_dead_cont)
    return times_eval, brier

# calculate the integrated brier score from the brier score
def get_ibs(perf_b):
    nsteps = len(perf_b[0])
    return np.sum((perf_b[1][:(nsteps-1)]+perf_b[1][1:])*(perf_b[0][1:]-perf_b[0][:(nsteps-1)])/2)
