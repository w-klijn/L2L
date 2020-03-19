import matplotlib.pyplot as plt
from scipy.special import softmax


def spike_plot(spikes, title, idx='', show=False):
    events = spikes["senders"]
    times = spikes["times"]

    plt.figure()
    plt.scatter(times, events, s=0.1)
    plt.title(title)
    plt.xlabel("time in ms")
    plt.ylabel("GID of spikes")
    if show:
        plt.show()
    else:
        plt.savefig('sp_input_{}.eps'.format(idx), format='eps')


def plot_data(idx, mean_ca_e, mean_ca_i, total_connections_e,
              total_connections_i):
    fig, ax1 = plt.subplots()
    ax1.plot(mean_ca_e, 'r',
             label='Ca Concentration Excitatory Neurons', linewidth=2.0)
    ax1.plot(mean_ca_i, 'b',
             label='Ca Concentration Inhibitory Neurons', linewidth=2.0)
    # ax1.set_ylim([0, 0.275])
    ax1.set_xlabel("Time in [s]")
    ax1.set_ylabel("Ca concentration")
    ax2 = ax1.twinx()
    ax2.plot(total_connections_e, 'm',
             label='Excitatory connections', linewidth=2.0, linestyle='--')
    ax2.plot(total_connections_i, 'k',
             label='Inhibitory connections', linewidth=2.0, linestyle='--')
    ax2.set_ylabel("Connections")
    plt.savefig('sp_mins_bulk_{}.eps'.format(idx), format='eps')
    plt.close()


def plot_data_out(idx, mean_ca_e_out, mean_ca_i_out):
    fig, ax1 = plt.subplots()
    ax1.plot(mean_ca_e_out[1], 'r',
             label='Ca Concentration Excitatory Neurons', linewidth=2.0)
    ax1.plot(mean_ca_i_out[1], 'b',
             label='Ca Concentration Inhibitory Neurons', linewidth=2.0)
    # ax1.set_ylim([0, 0.275])
    ax1.set_xlabel("Time in [s]")
    ax1.set_ylabel("Ca concentration")
    plt.savefig('sp_mins_out_{}.eps'.format(idx), format='eps')
    plt.close()


def plot_output(idx, mean_ca_e_out):
    plt.clf()
    plt.plot(softmax([mean_ca_e_out[j][-1] for j in range(10)]), '.')
    plt.savefig('output_{}.eps'.format(idx), format='eps')
