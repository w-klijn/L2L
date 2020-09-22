from scipy.special import softmax

import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

sns.set_context("paper", font_scale=1.5, rc={
    "lines.linewidth": 2., "grid.linewidth": 0.1})
sns.set(style="darkgrid")
sns.set_color_codes("dark")


def plot_image(image, random_id, iteration, path, save=True):
    if not save:
        return
    plottable_image = np.reshape(image, (28, 28))
    with sns.axes_style("white"):
        plt.imshow(plottable_image, cmap='gray_r')
        plt.title('Index: {}'.format(random_id))
        save_path = os.path.join(path, 'normal_input{}.eps'.format(iteration))
        plt.savefig(save_path, format='eps')
        plt.close()


def spike_plot(spikes, title, gen_idx, idx='', show=False, save=True):
    """ Plots spiking activity in a dot plot """
    print('Gen {} idx {}'.format(gen_idx, idx))
    events = spikes["senders"]
    times = spikes["times"]

    plt.figure()
    plt.scatter(times, events, s=0.1)
    plt.title(title)
    plt.xlabel("time in ms")
    plt.ylabel("GID of spikes")
    if show:
        plt.show()
    elif save:
        plt.savefig('sp_{}_{}_{}.eps'.format(
            title, gen_idx, idx), format='eps')
    plt.close()


def plot_fr(idx, mean_ca_e, mean_ca_i, save=True):
    """ Plots firing rate """
    if not save:
        return
    fig, ax1 = plt.subplots()
    ax1.plot(mean_ca_e, 'r',
             label='Ex.', linewidth=2.0)
    ax1.plot(mean_ca_i, 'b',
             label='Inh.', linewidth=2.0)
    # ax1.set_ylim([0, 0.275])
    ax1.set_xlabel("Time in [s]")
    ax1.set_ylabel("Firing Rate [Hz]")
    plt.savefig('sp_mins_bulk_{}.eps'.format(idx), format='eps')
    plt.close()


def plot_data_total(idx, mean_ca_e, mean_ca_i, total_connections_e,
                    total_connections_i):
    fig, ax1 = plt.subplots()
    ax1.plot(mean_ca_e, 'r',
             label='Ex.', linewidth=2.0)
    ax1.plot(mean_ca_i, 'b',
             label='Inh.', linewidth=2.0)
    # ax1.set_ylim([0, 0.275])
    ax1.set_xlabel("Time in [s]")
    ax1.set_ylabel("Firing Rate [Hz]")
    ax2 = ax1.twinx()
    ax2.plot(total_connections_e, 'm',
             label='Ex.', linewidth=2.0, linestyle='--')
    ax2.plot(total_connections_i, 'k',
             label='Inh.', linewidth=2.0, linestyle='--')
    ax2.set_ylabel("Firing Rate [Hz]")
    plt.savefig('sp_mins_bulk_{}.eps'.format(idx), format='eps')
    plt.close()


def plot_data_out(idx, mean_ca_e_out, mean_ca_i_out):
    fig, axes = plt.subplots(5, 2, sharex=True, sharey=True)
    print(mean_ca_e_out)
    for i in range(5):
        for j in range(2):
            # FIXME iteration is only up to 5
            axes[i][j].plot(mean_ca_e_out[i], 'r',
                            label='Ex.', linewidth=2.0)
            axes[i][j].plot(mean_ca_i_out[i], 'b',
                            label='Inh.', linewidth=2.0)
    # ax1.set_ylim([0, 0.275])
    plt.xlabel("Time in [s]")
    plt.ylabel("Firing Rate [Hz]")
    plt.savefig('sp_mins_out_{}.eps'.format(idx), format='eps')
    plt.close()


def plot_fr_comparison(c0, c1, title='fr_comparison.eps'):
    fig, (ax1, ax2) = plt.subplots(2)
    for c in c0:
        ax1.plot(c, color='r', label='0')
    for c in c1:
        ax1.plot(c, color='b', label='1')
    ax2.plot(c0.mean(0), color='r', label='0')
    ax2.plot(c1.mean(0), color='b', label='1')
    ax1.set_title('Firing activity')
    ax2.set_title('Mean firing activity')
    plt.ylabel('Firing rate in [Hz]')
    fig.savefig('fr_comparison.eps', format='eps')
    plt.close()


def plot_output(idx, mean_ca_e_out):
    plt.clf()
    plt.plot(softmax([mean_ca_e_out[j][-1] for j in range(10)]), '.')
    plt.savefig('output_{}.eps'.format(idx), format='eps')


def plot_data(idx, mean_ca_e, mean_ca_i, total_connections_e,
              total_connections_i):
    fig, ax1 = plt.subplots()
    ax1.plot(mean_ca_e, 'r',
             label='Ex.', linewidth=2.0)
    ax1.plot(mean_ca_i, 'b',
             label='Inh.', linewidth=2.0)
    # ax1.set_ylim([0, 0.275])
    ax1.set_xlabel("Time in [s]")
    ax1.set_ylabel("Firing Rate [Hz]")
    ax2 = ax1.twinx()
    ax2.plot(total_connections_e, 'm',
             label='Ex.', linewidth=2.0, linestyle='--')
    ax2.plot(total_connections_i, 'k',
             label='Inh.', linewidth=2.0, linestyle='--')
    ax2.set_ylabel("Firing Rate [Hz]")
    plt.savefig('sp_mins_bulk_{}.eps'.format(idx), format='eps')
    plt.close()
