#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
import matplotlib.pylab as plt
from matplotlib.widgets import Slider, RadioButtons
from matplotlib.colors import LogNorm
import numpy as np
import sys
import scipy.optimize as so

if __name__ == "__main__":
    data = np.genfromtxt(sys.argv[1], filling_values=[0, 0, 0, 0, 0])
    bin_count = int(sys.argv[2])

    xs = data.T[0]
    ys = data.T[1]

    v0 = data.T[3]
    v1 = data.T[4]

    v0_min = v0.min()
    v0_max = v0.max()

    f, ax = plt.subplots()
    ax.hist2d(xs, ys, bins=bin_count)

    def update_plot():
        global v0_min, v0_max, ax
        flag = [v0_min < v0[i] < v0_max for i in xrange(len(v0))]
        xs1 = np.array([xs[i] for i in xrange(len(v0)) if flag[i]])
        ys1 = np.array([ys[i] for i in xrange(len(v0)) if flag[i]])
        ax.hist2d(xs1, ys1, bins=bin_count)

    min_slider = Slider(plt.axes([0.1, 0.92, 0.8, 0.03]), 'v0_min', v0.min(), 200, valinit=v0.min())
    max_slider = Slider(plt.axes([0.1, 0.95, 0.8, 0.03]), 'v0_max', v0.min(), 200, valinit=200)
    def update_sliders(val):
        global v0_min, v0_max
        v0_min = min_slider.val
        v0_max = max_slider.val
        update_plot()
        plt.draw()

    min_slider.on_changed(update_sliders)
    max_slider.on_changed(update_sliders)

    plt.show()
