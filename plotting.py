# %%
from matplotlib import pyplot as plt
import scipy
import scipy.io
from plotting_utils import apply_settings, plot_cov_ellipse2d

import numpy as np

# from run_joyride import (
#      K, Ts, Xgt, Z, peak_pos_deviation, peak_vel_deviation,
#      x_hat, posRMSE, velRMSE, prob_hat,
#      NEESpos, ANEESpos, NEESvel, ANEESvel, NEES, ANEES,
#      CI2, CI4, CI2K, CI4K, confprob,
#  )

# %%
apply_settings()

# %% plot measurements close to the trajectory


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
        plt.show(block=False)
        

# trajectory
def plot_traj(  
        x_hat, 
        posRMSE, 
        velRMSE, 
        prob_hat, 
        peak_pos_deviation, 
        peak_vel_deviation
        ):
    
        fig3, axs3 = plt.subplots(1, 2, num=3, clear=True)
        axs3[0].plot(*x_hat.T[:2], label=r"$\hat x$")
        axs3[0].plot(*Xgt.T[:2], label="$x$")
        axs3[0].set_title(
            f"RMSE(pos, vel) = ({posRMSE:.3f}, {velRMSE:.3f})\npeak_dev(pos, vel)= ({peak_pos_deviation:.3f}, {peak_vel_deviation:.3f})"
        )
        axs3[0].axis("equal")
        # probabilities
        axs3[1].plot(np.cumsum(Ts), prob_hat)
        axs3[1].set_ylim([0, 1])
        axs3[1].set_ylabel("mode probability")
        axs3[1].set_xlabel("time")
        


# %% NEES
def plot_NEES_CI(
        NEESpos,
        ANEESpos,
        NEESvel,
        ANEESvel,
        NEES,
        ANEES):
    
        fig4, axs4 = plt.subplots(3, sharex=True, num=4, clear=True)
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
        
        print(f"ANEESpos = {ANEESpos:.2f} with CI = [{CI2K[0]:.2f}, {CI2K[1]:.2f}]")
        print(f"ANEESvel = {ANEESvel:.2f} with CI = [{CI2K[0]:.2f}, {CI2K[1]:.2f}]")
        print(f"ANEES = {ANEES:.2f} with CI = [{CI4K[0]:.2f}, {CI4K[1]:.2f}]")
        


# %% errors
def plot_errors(
        CI2,
        CI4,
        CI2K,
        CI4K,
        confprob,
        ):
    
        fig5, axs5 = plt.subplots(2, num=5, clear=True)
        axs5[0].plot(np.cumsum(Ts), np.linalg.norm(x_hat[:, :2] - Xgt[:, :2], axis=1))
        axs5[0].set_ylabel("position error")
        
        axs5[1].plot(np.cumsum(Ts), np.linalg.norm(
            x_hat[:, 2:4] - Xgt[:, 2:4], axis=1))
        axs5[1].set_ylabel("velocity error")
        

plt.show(block=False)

