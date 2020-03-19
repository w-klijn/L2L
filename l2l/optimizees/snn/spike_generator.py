import numpy as np

import matplotlib.pyplot as plt

__author__ = 'jacob'


def gen(input_img, n_neuron, time_step=1.0, name='bellec'):
    '''
    Function to call the various spike generator
    Input:
        input_img: linearized pixels of the MNIST number
        n_neuron: The number of input neurons, the last will be used for readout spike
        time_step: Difference in time steps between neuron firings
    '''
    if name == 'bellec':
        dist_spikes, t_spikes = bellec_spikes(input_img, n_neuron, time_step)
    return dist_spikes, t_spikes


# GENERATORS FOR THRESHOLDS
def _lin_gen(n_neurons, input_px):
    return np.linspace(np.min(input_px), np.max(input_px), n_neurons,
                       endpoint=False)


# GENERATORS FOR SPIKES
def tavanaei(input_img, n_neuron, time_step):
    '''
    Encode the binary spikes of the neurons into spike trains,
    combine some K trains into one neuron and encode.
    Input:
        input_img: BINARY pixel data of MNIST number
        n_neuron: Number of Neurons to encode this into
        time_step: Difference in times between neuron firings
    '''
    pass


def greyvalue(input_img, min_rate, max_rate):
    '''
    Encode the brightness of the neuron into firing probabilities
    of the input neurons
    Input:
        input_img: Pixel data of MNIST number
        min_rate: Rate corresponding to a pixel value of 0
        max_rate: Rate corresponding to a pixel value of 255
    '''
    max_value = input_img.max()
    # rates = np.zeros(len(input_img))
    rates = input_img * (max_rate - min_rate) / max_value + min_rate
    return rates


def greyvalue_sequential(input_img, start_time, end_time, min_rate, max_rate):
    '''
    Encode the brightness of the neuron into firing probabilities
    of the input neurons
    Input:
        input_img: Pixel data of MNIST number
        min_rate: Rate corresponding to a pixel value of 0
        max_rate: Rate corresponding to a pixel value of 255
    '''
    max_value = input_img.max()
    rates = np.zeros(len(input_img))
    start_times = np.linspace(start_time, end_time, len(input_img),
                              endpoint=True)
    end_times = start_times + (start_times[1] - start_times[0])
    for ii in range(len(input_img)):
        rates[ii] = input_img[ii] / max_value * (
                max_rate - min_rate) + min_rate
    return rates, start_times, end_times


def bellec_spikes(input_img, n_neuron, time_step=1.0, threshhold_gen=_lin_gen):
    '''
    Each neuron is associated with a threshhold value, when this threshhold is crossed in
    the transition from pixel ii to ii+1, the neuron fires at time ii.

    Input:
        input_img: Linearized pixels of the MNIST number, expected to x-folding
        n_neuron: The number of input neurons, the last will be used for the readout spike
        time_step: Difference in time steps between neuron firings
        threshhold_gen: Generator function for the threshhold values
    https://arxiv.org/abs/1803.09574
    '''
    spike_distribution = np.zeros((len(input_img), n_neuron))
    spike_end_offset = time_step * 10;

    neuron_threshhold = threshhold_gen(n_neuron - 1, input_img)
    spike_times = np.arange(start=0.0, stop=
    time_step * len(input_img), step=time_step)
    spike_times[-1] += spike_end_offset

    for ii in range(len(input_img) - 1):
        for jj in range(n_neuron - 1):
            if input_img[ii + 1] > neuron_threshhold[jj] >= input_img[ii]:
                spike_distribution[ii][jj] = 1
            elif input_img[ii + 1] < neuron_threshhold[jj] <= input_img[ii]:
                spike_distribution[ii][jj] = 1
    spike_distribution[-1][-1] = 1

    return spike_distribution, spike_times


def plot(dist_spikes, t_spikes):
    times = np.array([])
    spikes = np.array([])
    for ii in range(len(t_spikes)):
        spikes = np.append(spikes, np.nonzero(dist_spikes[ii]))
        times = np.append(times, np.array(
            [t_spikes[ii]] * np.count_nonzero(dist_spikes[ii])))
    plt.scatter(times, spikes, s=0.1)
    plt.ylabel("Input Channel")
    plt.xlabel("Time in ms")
    plt.title("Raw Input Spike data")
    plt.show(block=False)


def _test_bellec():
    test_arr = [1.0, 0.0, 0.0, 0.5];
    n_neuron = 11
    test_spikes, test_spike_times = gen(test_arr, n_neuron, name='bellec');
    plot(test_spikes, test_spike_times)
    valid = True
    if not (np.sum(test_spikes[-1, :]) == 1 \
            and np.sum(test_spikes[1, :]) == 0):
        raise Exception('FAILED: bellec_spikes')
        valid = False
    if not np.all((test_spikes.shape == (len(test_arr), n_neuron))):
        raise Exception('FAILED: bellec_spikes')
        valid = False
    if valid: print('SUCCESS: bellec_spikes')


def _test_lin_gen():
    valid = True
    if not np.all(_lin_gen(2, [0, 1]) == [0.0, 0.5]):
        raise Exception('FAILED: linear_generator')
        valid = False
    if valid: print('SUCCESS: linear_generator')


def _test_gen():
    pass


if __name__ == '__main__':
    _test_bellec()
    _test_lin_gen()
