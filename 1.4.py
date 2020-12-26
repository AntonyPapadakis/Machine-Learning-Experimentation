import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt


def x_to_phi(x):
    # get x and generate the corresponding 6-dim vector phi(x)
    phi = np.array([1, x, x**2, x**3, x**4, x**5])
    return phi

def generate_y(x, th, s_n, m_n):
    # generate a y for a specific x
    # use the true model and add noise n
    phi = x_to_phi(x)
    n = np.random.normal(m_n, s_n)
    y = phi @ th + n
    return y


def m_th_y(th_0, s_n, s_th, phis, ys):
    # find mu theta given y
    id = np.identity(th_0.shape[0])
    temp1 = 1 / s_n * inv((1 / s_th) * id + (1 / s_n) * phis.T @ phis)
    temp2 = phis.T @ (ys - phis @ th_0)
    res = th_0 + temp1 @ temp2
    return res

def get_m_y(x, m_th_y):
    phi = x_to_phi(x)
    m_y = phi.T @ m_th_y
    return m_y


def get_s_y_sq(x, s_n, s_th, phis):
    phi = x_to_phi(x)
    temp1 = s_n * s_th * phi.T
    temp2 = inv(s_n * np.identity(phis.shape[1]) + s_th * phis.T @ phis)
    res = s_n + (temp1 @ temp2) @ phi
    return res

np.random.seed(0)
theta_true = np.array([0.2, -1, 0.9, 0.7, 0, -0.2])

s_ns = [0.05, 0.15]
for s_n in s_ns:
    N = 20

    s_th = 0.1
    m_n = 0
    # form training set
    phis = np.zeros((N, theta_true.shape[0]))
    ys = np.zeros(N)
    x_train = np.linspace(0,2, N)
    for i in range(0, N):
        phi = x_to_phi(x_train[i])
        y = generate_y(x_train[i], theta_true, s_n, m_n)
        phis[i, :] = phi
        ys[i] = y

    # perform Bayesian Inference
    # find mu theta given y
    mu_y_th = m_th_y(theta_true, s_n, s_th, phis, ys)

    # create test set
    N = 20
    x_test = np.zeros(N)
    true_y = np.zeros(N)
    pred_y = np.zeros(N)
    err_y = np.zeros(N)
    for i in range(0, N):
        x = np.random.uniform(0,2)
        x_test[i] = x
        pred_y[i] = get_m_y(x_test[i], mu_y_th)
        err_y[i] = get_s_y_sq(x, s_n, s_th, phis)

    # generate a smooth true model with linspace
    x_for_true = np.linspace(0,2, 10000)
    true_y = np.zeros(x_for_true.shape[0])
    for i, x in enumerate(x_for_true):
        true_y[i] = generate_y(x, theta_true, 0.0, m_n)


    # plot results
    plt.title("Sigma noise: %.2f, Number of training points: %d" % (s_n, N))
    plt.scatter(x_for_true, true_y, color='red', marker='.', s=1)
    plt.errorbar(x_test, pred_y, yerr=err_y, fmt='o')
    plt.savefig("1.4_%s_%s.png" % (s_n, N))
    plt.show()



