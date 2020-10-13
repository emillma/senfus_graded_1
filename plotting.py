# %%
from matplotlib import pyplot as plt
import scipy
import scipy.io
from plotting_utils import apply_settings, plot_cov_ellipse2d
import numpy as np
from matplotlib import cm


def plot_measurements(
        K,
        Ts,
        Xgt,
        Z
):

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

# trajectory


def plot_traj(
        Ts,
        Xgt,
        x_hat,
        Z,
        gated_list,
        posRMSE,
        velRMSE,
        prob_hat,
        peak_pos_deviation,
        peak_vel_deviation
):

    fig3, axs3 = plt.subplots(1, 2, num=3, clear=True)
    gated_list = [
        'g' if item else 'r' for sublist in gated_list for item in sublist]
    axs3[0].scatter(*np.vstack(Z).T, color=gated_list, s=6)
    for i in range(x_hat.shape[0]-1):
        slice_tmp = slice(i, i+2)
        axs3[0].plot(*(x_hat[slice_tmp, :2].T),
                     color=cm.cool(prob_hat[i, 0]), linewidth=3)
        axs3[1].plot(np.cumsum(Ts)[slice_tmp], prob_hat[slice_tmp, 0],
                     color=cm.cool(prob_hat[i, 0]))

    axs3[0].plot(*Xgt.T[:2], label="$x$", color="C2")
    axs3[0].set_title(
        f"RMSE(pos, vel) = ({posRMSE:.3f}, {velRMSE:.3f})\npeak_dev(pos, vel)= ({peak_pos_deviation:.3f}, {peak_vel_deviation:.3f})"
    )
    axs3[0].axis("equal")
    # axs3[0].legend()
    # probabilities
    axs3[1].set_ylim([0, 1])
    axs3[1].set_ylabel("mode probability")
    axs3[1].set_xlabel("time")


# %% NEES
def plot_NEES_CI(
        Ts,
        NEESpos,
        ANEESpos,
        NEESvel,
        ANEESvel,
        NEES,
        ANEES,
        CI2,
        CI4,
        CI2K,
        CI4K,
        confprob):

    fig4, axs4 = plt.subplots(3, sharex=True, num=4, clear=True)
    for ax in axs4:
        ax.set_yscale('log')
    axs4[0].plot(np.cumsum(Ts), NEESpos)
    axs4[0].plot([0, sum(Ts)], np.repeat(CI2[None], 2, 0), "--r")
    axs4[0].set_ylabel("NEES pos")
    inCIpos = np.mean((CI2[0] <= NEESpos) * (NEESpos <= CI2[1]))
    axs4[0].set_title(f"{inCIpos*100:.1f}% inside {confprob*100:.1f}% CI")

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

    print(
        f"ANEESpos = {ANEESpos:.2f} with CI = [{CI2K[0]:.2f}, {CI2K[1]:.2f}]")
    print(
        f"ANEESvel = {ANEESvel:.2f} with CI = [{CI2K[0]:.2f}, {CI2K[1]:.2f}]")
    print(f"ANEES = {ANEES:.2f} with CI = [{CI4K[0]:.2f}, {CI4K[1]:.2f}]")


# %% errors
def plot_errors(
        Ts,
        Xgt,
        x_hat,
        CI2,
        CI4,
        CI2K,
        CI4K,
        confprob,
):

    fig5, axs5 = plt.subplots(2, num=5, clear=True)
    axs5[0].plot(np.cumsum(Ts), np.linalg.norm(
        x_hat[:, :2] - Xgt[:, :2], axis=1))
    axs5[0].set_ylabel("position error")

    axs5[1].plot(np.cumsum(Ts), np.linalg.norm(
        x_hat[:, 2:4] - Xgt[:, 2:4], axis=1))
    axs5[1].set_ylabel("velocity error")


def plot_NIS_NEES_model_specific(
        Ts,
        NIS_CV_list,
        NIS_CT_list,
        NEES_CV_list,
        NEES_CT_list,
        confprob):

    fig6, axs6 = plt.subplots(4, sharex=True, num=6, clear=True)
    for ax in axs6:
        ax.set_yscale('log')
    Ts_list = [np.cumsum(Ts)[k] for (k, data) in NIS_CV_list]
    NIS_data = [data for (k, data) in NIS_CV_list]

    CI4 = np.array(scipy.stats.chi2.interval(confprob, 4))

    axs6[0].plot(Ts_list, NIS_data)
    axs6[0].plot([0, sum(Ts)], np.repeat(CI4[None], 2, 0), "--r")
    axs6[0].set_ylabel("NIS CV")
    inCIpos = np.mean((CI4[0] <= NIS_data) * (NIS_data <= CI4[1]))
    axs6[0].set_title(
        f"NIS CV, {inCIpos*100:.1f}% inside {confprob*100:.1f}% CI")

    Ts_list = [np.cumsum(Ts)[k] for (k, data) in NIS_CT_list]
    NIS_data = [data for (k, data) in NIS_CT_list]

    axs6[1].plot(Ts_list, NIS_data)
    axs6[1].plot([0, sum(Ts)], np.repeat(CI4[None], 2, 0), "--r")
    axs6[1].set_ylabel("NIS CT")
    inCIpos = np.mean((CI4[0] <= NIS_data) * (NIS_data <= CI4[1]))
    axs6[1].set_title(
        f"NIS CT, {inCIpos*100:.1f}% inside {confprob*100:.1f}% CI")

    Ts_list = [np.cumsum(Ts)[k] for (k, data) in NEES_CV_list]
    NEES_data = [data for (k, data) in NEES_CV_list]

    axs6[2].plot(Ts_list, NEES_data)
    axs6[2].plot([0, sum(Ts)], np.repeat(CI4[None], 2, 0), "--r")
    axs6[2].set_ylabel("NEES CV")
    inCIpos = np.mean((CI4[0] <= NEES_data) * (NEES_data <= CI4[1]))
    axs6[2].set_title(
        f"NEES CV, {inCIpos*100:.1f}% inside {confprob*100:.1f}% CI")

    Ts_list = [np.cumsum(Ts)[k] for (k, data) in NEES_CT_list]
    NEES_data = [data for (k, data) in NEES_CT_list]

    axs6[3].plot(Ts_list, NEES_data)
    axs6[3].plot([0, sum(Ts)], np.repeat(CI4[None], 2, 0), "--r")
    axs6[3].set_ylabel("NEES CT")
    inCIpos = np.mean((CI4[0] <= NEES_data) * (NEES_data <= CI4[1]))
    axs6[3].set_title(
        f"NEES CT, {inCIpos*100:.1f}% inside {confprob*100:.1f}% CI")

    # axs4[1].plot(np.cumsum(Ts), NEESvel)
    # axs4[1].plot([0, sum(Ts)], np.repeat(CI2[None], 2, 0), "--r")
    # axs4[1].set_ylabel("NEES vel")
    # inCIvel = np.mean((CI2[0] <= NEESvel) * (NEESvel <= CI2[1]))
    # axs4[1].set_title(f"{inCIvel*100:.1f}% inside {confprob*100:.1f}% CI")

    # axs4[2].plot(np.cumsum(Ts), NEES)
    # axs4[2].plot([0, sum(Ts)], np.repeat(CI4[None], 2, 0), "--r")
    # axs4[2].set_ylabel("NEES")
    # inCI = np.mean((CI2[0] <= NEES) * (NEES <= CI2[1]))
    # axs4[2].set_title(f"{inCI*100:.1f}% inside {confprob*100:.1f}% CI")

    # print(
    #     f"ANEESpos = {ANEESpos:.2f} with CI = [{CI2K[0]:.2f}, {CI2K[1]:.2f}]")
    # print(
    #     f"ANEESvel = {ANEESvel:.2f} with CI = [{CI2K[0]:.2f}, {CI2K[1]:.2f}]")
    # print(f"ANEES = {ANEES:.2f} with CI = [{CI4K[0]:.2f}, {CI4K[1]:.2f}]")


def get_rotation_variance(Xgt):
    vel = Xgt[:, 2:]
    unit_vectors = vel / np.linalg.norm(vel, axis=1)[:, None]
    dot_product = np.sum(unit_vectors[1:] * unit_vectors[:-1], axis=1)
    angle = np.arccos(dot_product)
    return (f"Average rotation variance is"
            f"{np.var(angle[np.where(np.isfinite(angle))])/np.pi}*pi")
