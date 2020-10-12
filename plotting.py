# %%
from matplotlib import pyplot as plt
import scipy
import scipy.io
from plotting_utils import apply_settings, plot_cov_ellipse2d
import numpy as np

from run_joyride import (
    K, Ts, Xgt, Z,
    x_hat, posRMSE, velRMSE, prob_hat,
    NEESpos, ANEESpos, NEESvel, ANEESvel, NEES, ANEES,
    CI2, CI4, CI2K, CI4K, confprob,
)
# %%
apply_settings()

# %% plot measurements close to the trajectory
fig1, ax1 = plt.subplots(num=1, clear=True)

Z_plot_data = np.empty((0, 2), dtype=float)
plot_measurement_distance = np.inf
for Zk, xgtk in zip(Z, Xgt):
    to_plot = np.linalg.norm(
        Zk - xgtk[None:2], axis=1) <= plot_measurement_distance
    Z_plot_data = np.append(Z_plot_data, Zk[to_plot], axis=0)

ax1.scatter(*Z_plot_data.T, color="C1")
ax1.plot(*Xgt.T[:2], color="C0", linewidth=1.5)
ax1.set_title("True trajectory and the nearby measurements")
plt.show(block=False)

# trajectory
fig3, axs3 = plt.subplots(1, 2, num=3, clear=True)
axs3[0].plot(*x_hat.T[:2], label=r"$\hat x$")
axs3[0].plot(*Xgt.T[:2], label="$x$")
axs3[0].set_title(
    f"RMSE(pos, vel) = ({posRMSE:.3f}, {velRMSE:.3f})\npeak_dev(pos, vel) "
    "= ({peak_pos_deviation:.3f}, {peak_vel_deviation:.3f})"
)
axs3[0].axis("equal")
# probabilities
axs3[1].plot(np.cumsum(Ts), prob_hat)
axs3[1].set_ylim([0, 1])
axs3[1].set_ylabel("mode probability")
axs3[1].set_xlabel("time")

# %% NEES
fig4, axs4 = plt.subplots(3, sharex=True, num=4, clear=True)
axs4[0].plot(np.cumsum(Ts), NEESpos)
axs4[0].plot([0, sum(Ts)], np.repeat(CI2[None], 2, 0), "--r")
axs4[0].set_ylabel("NEES pos")
inCIpos = np.mean((CI2[0] <= NEESpos) * (NEESpos <= CI2[1]))
axs4[0].set_title("{inCIpos*100:.1f}% inside {confprob*100:.1f}% CI")

axs4[1].plot(np.cumsum(Ts), NEESvel)
axs4[1].plot([0, sum(Ts)], np.repeat(CI2[None], 2, 0), "--r")
axs4[1].set_ylabel("NEES vel")
inCIvel = np.mean((CI2[0] <= NEESvel) * (NEESvel <= CI2[1]))
axs4[1].set_title(f"{inCIvel*100:.1f}% inside {confprob*100:.1f}% CI")

axs4[2].plot(np.cumsum(Ts), NEES)
axs4[2].plot([0, sum(Ts)], np.repeat(CI4[None], 2, 0), "--r")
axs4[2].set_ylabel("NEES")
inCI = np.mean((CI2[0] <= NEES) * (NEES <= CI2[1]))
axs4[2].set_title(f"{inCI*100:.1f}% inside {confprob*100:.1f}% CI")

print(f"ANEESpos = {ANEESpos:.2f} with CI = [{CI2K[0]:.2f}, {CI2K[1]:.2f}]")
print(f"ANEESvel = {ANEESvel:.2f} with CI = [{CI2K[0]:.2f}, {CI2K[1]:.2f}]")
print(f"ANEES = {ANEES:.2f} with CI = [{CI4K[0]:.2f}, {CI4K[1]:.2f}]")

# %% errors
fig5, axs5 = plt.subplots(2, num=5, clear=True)
axs5[0].plot(np.cumsum(Ts), np.linalg.norm(x_hat[:, :2] - Xgt[:, :2], axis=1))
axs5[0].set_ylabel("position error")

axs5[1].plot(np.cumsum(Ts), np.linalg.norm(
    x_hat[:, 2:4] - Xgt[:, 2:4], axis=1))
axs5[1].set_ylabel("velocity error")

# %% TBD: estimation "movie"

# play_estimation_movie = True
# mTL = 0.2  # maximum transparancy (between 0 and 1);
# plot_pause = 1  # lenght to pause between time steps;
# start_k = 0
# end_k = K
# plot_range = slice(start_k, end_k)  # the range to go through

# # %k = 31; assert(all([k > 1, k <= K]), 'K must be in proper range')
# fig6, axs6 = plt.subplots(1, 2, num=6, clear=True)
# mode_lines = [axs6[0].plot(np.nan, np.nan, color=f"C{s}")[0] for s in range(2)]
# meas_sc = axs6[0].scatter(np.nan, np.nan, color="r", marker="x")
# meas_sc_true = axs6[0].scatter(np.nan, np.nan, color="g", marker="x")
# min_ax = np.vstack(Z).min(axis=0)  # min(cell2mat(Z'));
# max_ax = np.vstack(Z).max(axis=0)  # max(cell2mat(Z'));
# axs6[0].axis([min_ax[0], max_ax[0], min_ax[1], max_ax[0]])

# for k, (Zk, pred_k, upd_k) in enumerate(
#     zip(
#         Z[plot_range],
#         tracker_predict_list[plot_range],
#         tracker_update_list[plot_range],
#         # true_association[plot_range],
#     ),
#     start_k,
# ):
#     (ax.cla() for ax in axs6)
#     pl = []
#     gated = tracker.gate(Zk, pred_k)
#     minG = 1e20 * np.ones(2)
#     maxG = np.zeros(2)
#     cond_upd_k = tracker.conditional_update(Zk[gated], pred_k)
#     beta_k = tracker.association_probabilities(Zk[gated], pred_k)
#     for s in range(2):
#         mode_lines[s].set_data = (
#             np.array([u.components[s].mean[:2]
#                       for u in tracker_update_list[:k]]).T,
#         )
#         axs6[1].plot(prob_hat[: (k - 1), s], color=f"C{s}")
#         for j, cuj in enumerate(cond_upd_k):
#             alpha = 0.7 * beta_k[j] * cuj.weights[s] + 0.3
#             upd_km1_s = tracker_update_list[k - 1].components[s]
#             pl.append(
#                 axs6[0].plot(
#                     [upd_km1_s.mean[0], cuj.components[s].mean[0]],
#                     [upd_km1_s.mean[1], cuj.components[s].mean[1]],
#                     "--",
#                     color=f"C{s}",
#                     alpha=alpha,
#                 )
#             )

#             pl.append(
#                 axs6[1].plot(
#                     [k - 1, k],
#                     [prob_hat[k - 1, s], cuj.weights[s]],
#                     color=f"C{s}",
#                     alpha=alpha,
#                 )
#             )
#             pl.append(
#                 plot_cov_ellipse2d(
#                     axs6[0],
#                     cuj.components[s].mean[:2],
#                     cuj.components[s].cov[:2, :2],
#                     edgecolor=f"C{s}",
#                     alpha=alpha,
#                 )
#             )

#         Sk = imm_filter.filters[s].innovation_cov([0, 0], pred_k.components[s])
#         pl.append(
#             plot_cov_ellipse2d(
#                 axs6[0],
#                 pred_k.components[s].mean[:2],
#                 Sk,
#                 n_sigma=tracker.gate_size,
#                 edgecolor=f"C{s}",
#             )
#         )
#         meas_sc.set_offsets(Zk)
#         pl.append(axs6[0].scatter(*Zk.T, color="r", marker="x"))
#     plt.gcf().canvas.draw_idle()
#     plt.gcf().canvas.start_event_loop(0.01)
# # %%
plt.show(block=True)
