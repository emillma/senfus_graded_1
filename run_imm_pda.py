# %% imports
from typing import List

import scipy
import scipy.io
import scipy.stats

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from gaussparams import GaussParams
from mixturedata import MixtureParameters
import dynamicmodels
import measurementmodels
import ekf
import imm
import pda
import estimationstatistics as estats

from plotting_utils import apply_settings, plot_cov_ellipse2d

from plotting import (plot_measurements, plot_traj, plot_NEES_CI, plot_errors,
                      plot_NIS_CV)

# %% plot config check and style setup

apply_settings()


# %% load data
filename_to_load = "data_for_imm_pda.mat"
loaded_data = scipy.io.loadmat(filename_to_load)
K = loaded_data["K"].item()
# Ts = loaded_data["Ts"].item()
Ts = [loaded_data["Ts"].item() for i in range(K)]
Xgt = loaded_data["Xgt"].T
Z = [zk.T for zk in loaded_data["Z"].ravel()]
true_association = loaded_data["a"].ravel()

# %% IMM-PDA

# THE PRESET PARAMETERS AND INITIAL VALUES WILL CAUSE TRACK LOSS!
# Some reasoning and previous exercises should let you avoid track loss.
# No exceptions should be generated if PDA works correctly with IMM,
# but no exceptions do not guarantee correct implementation.

# sensor
sigma_z = 3
clutter_intensity = 0.002
PD = 0.99
gate_size = 5

# dynamic models
sigma_a_CV = 0.3
sigma_a_CT = 0.1
sigma_omega = 0.002*np.pi


# markov chain
PI11 = 0.9
PI22 = 0.9

p10 = 0.9  # initvalue for mode probabilities

PI = np.array([[PI11, (1 - PI11)], [(1 - PI22), PI22]])
assert np.allclose(np.sum(PI, axis=1), 1), "rows of PI must sum to 1"

# not valid

mean_init = np.array([0, 0, 0, 0, 0])
cov_init = np.diag([50, 50, 1, 1, 0.1]) ** 2  # THIS WILL NOT BE GOOD
mode_probabilities_init = np.array([p10, (1 - p10)])
mode_states_init = GaussParams(mean_init, cov_init)
init_imm_state = MixtureParameters(
    mode_probabilities_init, [mode_states_init] * 2)

assert np.allclose(
    np.sum(mode_probabilities_init), 1
), "initial mode probabilities must sum to 1"

# make model
measurement_model = measurementmodels.CartesianPosition(sigma_z, state_dim=5)
dynamic_models: List[dynamicmodels.DynamicModel] = []
dynamic_models.append(dynamicmodels.WhitenoiseAccelleration(sigma_a_CV, n=5))
dynamic_models.append(dynamicmodels.ConstantTurnrate(sigma_a_CT, sigma_omega))
ekf_filters = []
ekf_filters.append(ekf.EKF(dynamic_models[0], measurement_model))
ekf_filters.append(ekf.EKF(dynamic_models[1], measurement_model))
imm_filter = imm.IMM(ekf_filters, PI)

tracker = pda.PDA(imm_filter, clutter_intensity, PD, gate_size)

# init_imm_pda_state = tracker.init_filter_state(init__immstate)


NEES = np.zeros(K)
NEESpos = np.zeros(K)
NEESvel = np.zeros(K)

tracker_update = init_imm_state
tracker_update_list = []
tracker_predict_list = []
tracker_estimate_list = []
# estimate
for k, (Zk, x_true_k) in enumerate(zip(Z, Xgt)):
    tracker_predict = tracker.predict(tracker_update, Ts[k])
    tracker_update = tracker.update(Zk, tracker_predict)

    # You can look at the prediction estimate as well
    tracker_estimate = tracker.estimate(tracker_update)

    NEES[k] = estats.NEES_indexed(
        tracker_estimate.mean, tracker_estimate.cov, x_true_k, idxs=np.arange(
            4)
    )

    NEESpos[k] = estats.NEES_indexed(
        tracker_estimate.mean, tracker_estimate.cov, x_true_k, idxs=np.arange(
            2)
    )
    NEESvel[k] = estats.NEES_indexed(
        tracker_estimate.mean, tracker_estimate.cov, x_true_k, idxs=np.arange(
            2, 4)
    )

    tracker_predict_list.append(tracker_predict)
    tracker_update_list.append(tracker_update)
    tracker_estimate_list.append(tracker_estimate)


x_hat = np.array([est.mean for est in tracker_estimate_list])
prob_hat = np.array([upd.weights for upd in tracker_update_list])

# calculate a performance metrics
poserr = np.linalg.norm(x_hat[:, :2] - Xgt[:, :2], axis=0)
velerr = np.linalg.norm(x_hat[:, 2:4] - Xgt[:, 2:4], axis=0)
posRMSE = np.sqrt(
    np.mean(poserr ** 2)
)  # not true RMSE (which is over monte carlo simulations)
velRMSE = np.sqrt(np.mean(velerr ** 2))
# not true RMSE (which is over monte carlo simulations)
peak_pos_deviation = poserr.max()
peak_vel_deviation = velerr.max()


# consistency
confprob = 0.9
CI2 = np.array(scipy.stats.chi2.interval(confprob, 2))
CI4 = np.array(scipy.stats.chi2.interval(confprob, 4))

confprob = confprob
CI2K = np.array(scipy.stats.chi2.interval(confprob, 2 * K)) / K
CI4K = np.array(scipy.stats.chi2.interval(confprob, 4 * K)) / K
ANEESpos = np.mean(NEESpos)
ANEESvel = np.mean(NEESvel)
ANEES = np.mean(NEES)

#Plots
# trajectory
# #Fig 1
plot_measurements(K, Ts, Xgt, Z)

# #Fig 3
plot_traj(Ts, Xgt, x_hat, Z, posRMSE, velRMSE, prob_hat,
              peak_pos_deviation, peak_vel_deviation
              )
# #Fig 4
plot_NEES_CI(Ts, NEESpos, ANEESpos, NEESvel, ANEESvel, NEES, ANEES,
              CI2, CI4, CI2K, CI4K, confprob)

#Fig 5 errors
plot_errors(Ts, Xgt, x_hat, CI2, CI4, CI2K, CI4K, confprob)


plt.show()
# %%
