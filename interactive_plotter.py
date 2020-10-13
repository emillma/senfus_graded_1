from matplotlib import pyplot as plt
from plotting_utils import apply_settings, plot_cov_ellipse2d
import sys


class Plotter():
    def __init__(self):
        self.fig, self, ax = plt.subplots(2)
        self.fig.canvas.mpl_connect('key_press_event',
                                    self.key_press_event)

    def key_press_event(self, event):
        if event.key == 'q':
            sys.exit("Closed from GUI")
            self._close()
        elif event.key == 'd':
            for ax in [self.az_ax, self.el_ax, self.ctrl_ax]:
                ylim = ax.get_ylim()
                ax.clear()
                ax.set_ylim(ylim)
            self.set_t_zero(-5)
