import numpy as np
from numba import jit

@jit(nopython=True)
def _get_stats(time_event, label_event):

    n_ind = time_event.shape[0]
    sort_index = np.argsort(time_event)
    times = np.empty(n_ind)
    events = np.empty(n_ind)
    counts = np.empty(n_ind)

    time_eval = time_event[sort_index[0]]
    i = 0
    j = 0
    while i<=(n_ind-1):
        count_event = 0
        count = 0
        # repeated times for indivuduals
        while i < n_ind and time_eval == time_event[sort_index[i]]:
            if label_event[sort_index[i]]:
                count_event += 1
            count += 1
            i += 1
        # register the unique time
        times[j] = time_eval
        # add number of events at this time
        events[j] = count_event
        # add number of individuals at this time
        counts[j] = count
        # increment j
        j += 1
        # assign the next time
        if i < n_ind:
            time_eval = time_event[sort_index[i]]
        else:
            break
    
    times = times[:j]
    events = events[:j]
    counts = counts[:j]

    censors = counts - events
    # at_risk = (n_ind - np.r_[0,np.cumsum(counts)])[:-1]
    at_risk = (n_ind - np.append(0,np.cumsum(counts)))[:-1]

    return times, events, censors, at_risk