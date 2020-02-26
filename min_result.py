import numpy as np
from scipy.optimize import minimize


def min_result(samples, label, k, cnt):

    result = []
    for i in range(k):
        idx = label == i
        sum_color = np.sum(cnt[idx])
        x0 = np.array([128, 128, 128])
        f = lambda x: np.sum(np.sum(np.abs(x-samples[idx, :]), axis=1)*cnt[idx]) / sum_color
        res = minimize(fun=f, x0=x0, method='Nelder-Mead')
        result.append(res.x)

    return np.array(result)




