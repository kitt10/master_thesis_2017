#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    scripts.script
    ~~~~~~~~~~~~~~

    Doing quick pythonic stuff.
"""

from scipy.io.wavfile import read as read_wav
from matplotlib import pyplot as plt, patches as mpl_patches, rcParams as mpl_params
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

MAX_INT16 = 32768
mpl_params['axes.labelsize'] = 24
mpl_params['xtick.labelsize'] = 20
mpl_params['ytick.labelsize'] = 20

if __name__ == '__main__':
    sampling_rate, signal = read_wav('../../lubos/data/wavs/SADB-2010-NTB/_data8kHz-cut/spk-1/spk-1-1.wav')
    signal = [float(s)/MAX_INT16 for s in signal]   # normalisation
    duration = float(len(signal))/sampling_rate
    win_len = 0.025
    win_shift = 0.01

    fig, ax = plt.subplots()
    ax.plot([float(s)/sampling_rate for s in range(len(signal))], signal, alpha=0.3)
    ax.add_patch(mpl_patches.Rectangle((1, -0.01), 0.1, 0.02, fill=False, color='red'))
    col = ('magenta', 'red', 'orange', 'maroon')
    for i in range(int(duration/win_shift)):
        ax.add_patch(mpl_patches.Rectangle((i*win_shift, -0.48), win_len, 0.96, fill=False, color=col[i%4], alpha=0.2))

    ax.set_xlim([0, duration])
    ax.set_xlabel('time [s]')
    ax.set_ylim([-0.5, 0.5])
    ax.set_ylabel('amplitude')
    ax.grid()

    axins = zoomed_inset_axes(ax, zoom=15, loc=4, borderpad=2)
    axins.set_yticks((),())
    axins.set_xlim([1, 1.1])
    axins.set_ylim([-0.01, 0.01])
    mark_inset(ax, axins, loc1=1, loc2=2, color='red')

    axins.plot([float(s)/sampling_rate for s in range(len(signal))], signal, alpha=0.3)
    for i in range(int(duration/win_shift)):
        axins.add_patch(mpl_patches.Rectangle((i*win_shift, -0.48), win_len, 0.96, fill=False, color=col[i%4]))

    axins.annotate(s='window size', xy=(1, 0), xytext=(1+win_len+0.0015, 0),
                   arrowprops=dict(arrowstyle='<->'), va='center', backgroundcolor='yellow', fontsize=16)
    axins.annotate(s='window shift', xy=(1, -0.005), xytext=(1+win_shift+0.0015, -0.005),
                   arrowprops=dict(arrowstyle='<->'), va='center', backgroundcolor='yellow', fontsize=16)

    plt.tight_layout()
    plt.show()