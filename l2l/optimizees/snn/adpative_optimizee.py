from collections import OrderedDict, namedtuple
from l2l.optimizees.optimizee import Optimizee
from l2l.optimizees.snn import spike_generator, visualize
from scipy.special import softmax

import json
import glob
import numpy as np
import nest
import os
import pandas as pd
import pickle

AdaptiveOptimizeeParameters = namedtuple(
    'AdaptiveOptimizeeParameters', ['seed', 'path',
                                    'record_spiking_firingrate',
                                    'save_plot'])


class AdaptiveOptimizee(Optimizee):
    def __init__(self, traj, parameters):
        super().__init__(traj)
        self.parameters = parameters
        seed = np.uint32(self.parameters.seed)

        self.random_state = np.random.RandomState(seed=seed)
        with open('/home/yegenoglu/Documents/toolbox/L2L/l2l/optimizees/snn/config.json') as jsonfile:
            self.config = json.load(jsonfile)
        seed = np.uint32(self.config['seed'])
        self.random_state = np.random.RandomState(seed=seed)
        self.t_sim = self.config['t_sim']
        self.input_type = self.config['input_type']
        # Resolution, simulation steps in [ms]
        self.dt = self.config['dt']
        self.neuron_model = self.config['neuron_model']
        self.gen_idx = traj.individual.generation
        self.ind_idx = traj.individual.ind_idx

        self.n_input_neurons = self.config['n_input']
        self.n_bulk_ex_neurons = self.config['n_bulk_ex']
        self.n_bulk_in_neurons = self.config['n_bulk_in']
        self.psc_e = self.config['psc_e']
        self.psc_i = self.config['psc_i']
        self.psc_ext = self.config['psc_ext']
        self.bg_rate = self.config['bg_rate']
        self.record_interval = self.config['record_interval']
        self.warm_up_time = self.config['warm_up_time']

        self.nodes_in = None
        self.nodes_e = None
        self.nodes_i = None
        self.pixel_rate_generators = None
        self.noise = None
        self.input_spike_detector = None
        self.bulks_detector_ex = None
        self.bulks_detector_in = None
        self.rates = None
        self.target_px = None
        self.weights_e = None
        self.weights_i = None

        self.total_connections_e = []
        self.total_connections_i = []
        self.mean_ca_e = []
        self.mean_ca_i = []
        self.target_labels = []
        self.random_ids = []

    def connect_network(self):
        self.prepare_network()
        # Do the connections
        self.connect_external_input(self.n_input_neurons)
        self.connect_input_spike_detectors()
        self.connect_internal_bulk()

        # save connection structure
        conns_e = nest.GetConnections(source=self._get_net_structure('e'))
        self.save_connections(conns_e, self.gen_idx, self.ind_idx,
                              path=self.parameters.path,
                              typ='e')
        conns_i = nest.GetConnections(source=self._get_net_structure('i'))
        self.save_connections(conns_i, self.gen_idx, self.ind_idx,
                              path=self.parameters.path,
                              typ='i')
        return len(conns_e), len(conns_i)

    def prepare_network(self):
        """  Helper functions to create the network """
        self.reset_kernel()
        self.create_nodes()
        self.create_synapses()
        self.create_input_spike_detectors()
        self.pixel_rate_generators = self.create_pixel_rate_generator(
            self.input_type)
        self.noise = nest.Create("poisson_generator")
        nest.PrintNetwork(depth=2)
        self.connect_external_input(self.n_input_neurons)
        self.connect_input_spike_detectors()
        self.connect_all_spike_detectors()
        self.connect_internal_bulk()
        self.connect_noise_bulk()

    def _get_net_structure(self, typ):
        if typ == 'e':
            return tuple(self.nodes_in + self.nodes_e)
        else:
            return tuple(self.nodes_in + self.nodes_i)

    def reset_kernel(self):
        nest.ResetKernel()
        nest.set_verbosity('M_ERROR')
        nest.SetKernelStatus({'resolution': self.dt,
                              'local_num_threads': 4,
                              'overwrite_files': True})

    def create_nodes(self):
        self.nodes_in = nest.Create(
            self.neuron_model, self.n_input_neurons)
        self.nodes_e = nest.Create(self.neuron_model, self.n_bulk_ex_neurons)
        self.nodes_i = nest.Create(self.neuron_model, self.n_bulk_in_neurons)
        # TODO add the output cluster
        # TODO enable commented region
        # nest.SetStatus(
        #     self.nodes_e, {'a': self.params['a'], 'b': self.params['b']})
        # nest.SetStatus(
        #     self.nodes_i, {'a': self.params['a'], 'b': self.params['b']})

    def create_input_spike_detectors(self, record_fr=True):
        self.input_spike_detector = nest.Create("spike_detector",
                                                params={"withgid": True,
                                                        "withtime": True})
        if record_fr:
            self.bulks_detector_ex = nest.Create("spike_detector",
                                                 params={"withgid": True,
                                                         "withtime": True,
                                                         "to_file": True,
                                                         "label": "bulk_ex",
                                                         "file_extension": "spikes"})
            self.bulks_detector_in = nest.Create("spike_detector",
                                                 params={"withgid": True,
                                                         "withtime": True,
                                                         "to_file": True,
                                                         "label": "bulk_in",
                                                         "file_extension": "spikes"})

    def create_pixel_rate_generator(self, input_type):
        if input_type == 'greyvalue':
            return nest.Create("poisson_generator",
                               self.n_input_neurons)
        elif input_type == 'bellec':
            return nest.Create("spike_generator",
                               self.n_input_neurons)
        elif input_type == 'greyvalue_sequential':
            n_img = self.n_input_neurons
            rates, starts, ends = spike_generator.greyvalue_sequential(
                self.target_px[n_img], start_time=0, end_time=783, min_rate=0,
                max_rate=10)
            self.rates = rates
            # FIXME changed to len(rates) from len(offsets)
            self.pixel_rate_generators = nest.Create(
                "poisson_generator", len(rates))

    @staticmethod
    def create_synapses():
        nest.CopyModel('static_synapse', 'random_synapse')
        nest.CopyModel('static_synapse', 'random_synapse_i')

    def create_spike_rate_generator(self, input_type):
        if input_type == 'greyvalue':
            return nest.Create("poisson_generator",
                               self.n_input_neurons)
        elif input_type == 'bellec':
            return nest.Create("spike_generator",
                               self.n_input_neurons)
        elif input_type == 'greyvalue_sequential':
            n_img = self.n_input_neurons
            rates, starts, ends = spike_generator.greyvalue_sequential(
                self.target_px[n_img], start_time=0, end_time=783, min_rate=0,
                max_rate=10)
            self.rates = rates
            # FIXME changed to len(rates) from len(offsets)
            self.pixel_rate_generators = nest.Create(
                "poisson_generator", len(rates))

    def connect_input_spike_detectors(self):
        nest.Connect(self.nodes_in, self.input_spike_detector)

    def connect_all_spike_detectors(self):
        # BULK
        # TODO? self.nodes_e[0:self.number_recorded_bulk_exc], self.bulksde)
        nest.Connect(self.nodes_e, self.bulks_detector_ex)
        nest.Connect(self.nodes_i, self.bulks_detector_in)

    def connect_noise_bulk(self):
        poisson_gen = nest.Create("poisson_generator", 1, {'rate': 10000.0}, )
        syn_dict = {"model": "static_synapse", "weight": 1}
        syn_dict_i = {"model": "static_synapse", "weight": 1}
        nest.Connect(poisson_gen, self.nodes_e, "all_to_all",
                     syn_spec=syn_dict)
        nest.Connect(poisson_gen, self.nodes_i, "all_to_all",
                     syn_spec=syn_dict_i)

    def connect_greyvalue_input(self):
        """ Connects input to bulk """
        syn_dict_e = {"model": "random_synapse",
                      'weight': {"distribution": "normal",
                                 "mu": self.psc_e,
                                 "sigma": 100.}}
        syn_dict_i = {"model": "random_synapse_i",
                      'weight': {"distribution": "normal",
                                 "mu": self.psc_i,
                                 "sigma": 100.}}
        nest.Connect(self.pixel_rate_generators, self.nodes_in, "one_to_one",
                     syn_spec=syn_dict_e)
        # connect input to bulk
        nest.Connect(self.nodes_in, self.nodes_e, "all_to_all",
                     syn_spec=syn_dict_e)
        nest.Connect(self.nodes_in, self.nodes_i, "all_to_all",
                     syn_spec=syn_dict_i)

    def connect_bellec_input(self):
        nest.Connect(self.pixel_rate_generators, self.nodes_in, "one_to_one")
        weights = {'distribution': 'uniform',
                   'low': self.psc_i, 'high': self.psc_e}
        syn_dict = {"model": "random_synapse", "weight": weights}
        conn_dict = {'rule': 'fixed_outdegree',
                     'outdegree': int(0.05 * self.n_bulk_ex_neurons)}
        nest.Connect(self.nodes_in, self.nodes_e,
                     conn_spec=conn_dict, syn_spec=syn_dict)
        conn_dict = {'rule': 'fixed_outdegree',
                     'outdegree': int(0.05 * self.n_bulk_in_neurons)}
        nest.Connect(self.nodes_in, self.nodes_i,
                     conn_spec=conn_dict, syn_spec=syn_dict)

    def clear_input(self):
        """
        Sets a very low rate to the input, for the case where no input is
        provided
        """
        generator_stats = [{'rate': 1.0} for _ in
                           range(self.n_input_neurons)]
        nest.SetStatus(self.pixel_rate_generators, generator_stats)

    def connect_internal_bulk(self):
        syn_dict_e = {"model": "random_synapse",
                      'weight': {"distribution": "normal",
                                 "mu": self.psc_e,
                                 "sigma": 100.}}
        syn_dict_i = {"model": "random_synapse_i",
                      'weight': {"distribution": "normal",
                                 "mu": self.psc_i,
                                 "sigma": 100.}}
        # Connect bulk
        conn_dict = {'rule': 'fixed_outdegree',
                     'outdegree': int(0.06 * self.n_bulk_ex_neurons)}
        nest.Connect(self.nodes_e, self.nodes_e, conn_dict,
                     syn_spec=syn_dict_e)
        conn_dict = {'rule': 'fixed_outdegree',
                     'outdegree': int(0.08 * self.n_bulk_in_neurons)}
        nest.Connect(self.nodes_e, self.nodes_i, conn_dict,
                     syn_spec=syn_dict_e)
        conn_dict = {'rule': 'fixed_outdegree',
                     'outdegree': int(0.1 * self.n_bulk_ex_neurons)}
        nest.Connect(self.nodes_i, self.nodes_e, conn_dict,
                     syn_spec=syn_dict_i)
        conn_dict = {'rule': 'fixed_outdegree',
                     'outdegree': int(0.08 * self.n_bulk_in_neurons)}
        nest.Connect(self.nodes_i, self.nodes_i, conn_dict,
                     syn_spec=syn_dict_i)

    def connect_external_input(self, n_img):
        nest.SetStatus(self.noise, {"rate": self.bg_rate})
        nest.Connect(self.noise, self.nodes_e, 'all_to_all',
                     {'weight': self.psc_ext, 'delay': 1.0})
        nest.Connect(self.noise, self.nodes_i, 'all_to_all',
                     {'weight': self.psc_ext, 'delay': 1.0})

        if self.input_type == 'bellec':
            self.connect_bellec_input()
        elif self.input_type == 'greyvalue':
            self.connect_greyvalue_input()
        # elif self.input_type == 'greyvalue_sequential':
        #     self.connect_greyvalue_sequential_input()

    def clear_spiking_events(self):
        nest.SetStatus(self.bulks_detector_ex, "n_events", 0)
        nest.SetStatus(self.bulks_detector_in, "n_events", 0)

    def record_fr(self, indx, gen_idx, save=True):
        """ Records firing rates """
        n_recorded_bulk_ex = self.n_bulk_ex_neurons
        n_recorded_bulk_in = self.n_bulk_in_neurons
        self.mean_ca_e.append(
            nest.GetStatus(self.bulks_detector_ex, "n_events")[
                0] * 1000.0 / (self.record_interval * n_recorded_bulk_ex))
        self.mean_ca_i.append(
            nest.GetStatus(self.bulks_detector_in, "n_events")[
                0] * 1000.0 / (self.record_interval * n_recorded_bulk_in))
        spikes = nest.GetStatus(self.bulks_detector_ex, keys="events")[0]
        visualize.spike_plot(spikes, "Bulk spikes",
                             idx=indx, gen_idx=gen_idx, save=save)

    def record_ca(self):
        ca_e = nest.GetStatus(self.nodes_e, 'Ca'),  # Calcium concentration
        self.mean_ca_e.append(np.mean(ca_e))
        ca_i = nest.GetStatus(self.nodes_i, 'Ca'),  # Calcium concentration
        self.mean_ca_i.append(np.mean(ca_i))

    def clear_records(self):
        self.mean_ca_i = []
        self.mean_ca_e = []
        # try:
        #     nest.SetStatus(self.input_spike_detector, {"n_events": 0})
        # except AttributeError as e:
        #     print(e)
        #     pass

    def record_connectivity(self):
        syn_elems_e = nest.GetStatus(self.nodes_e, 'synaptic_elements')
        syn_elems_i = nest.GetStatus(self.nodes_i, 'synaptic_elements')
        # self.total_connections_e.append(sum(neuron['Bulk_E_Axn']['z_connected']
        #                                     for neuron in syn_elems_e))
        # self.total_connections_i.append(sum(neuron['Bulk_I_Axn']['z_connected']
        #                                     for neuron in syn_elems_i))

    def set_external_input(self, iteration, train_px_one, target):
        train_px = train_px_one
        path = self.parameters.path
        save = self.parameters.save_plot
        random_id = np.random.randint(low=0, high=len(train_px))
        self.random_ids.append(random_id)
        # leave the target labels as strings, which will be easier to save in
        # a dictionary later on
        label = target[random_id]
        if isinstance(label, str):
            label = int(label)
        self.target_labels.append(label)
        image = train_px[random_id]
        # Save image for reference
        visualize.plot_image(image=image, random_id=random_id,
                             iteration=iteration, path=path, save=save)
        if self.input_type == 'greyvalue':
            rates = spike_generator.greyvalue(image,
                                              min_rate=1, max_rate=100)
            generator_stats = [{'rate': w} for w in rates]
            nest.SetStatus(self.pixel_rate_generators, generator_stats)
        elif self.input_type == 'greyvalue_sequential':
            rates = spike_generator.greyvalue_sequential(image,
                                                         min_rate=1,
                                                         max_rate=100,
                                                         start_time=0,
                                                         end_time=783)
            generator_stats = [{'rate': w} for w in rates]
            nest.SetStatus(self.pixel_rate_generators, generator_stats)
        else:
            train_spikes, train_spike_times = spike_generator.bellec_spikes(
                train_px[random_id], self.n_input_neurons, self.dt)
            for ii, ii_spike_gen in enumerate(self.pixel_rate_generators):
                iter_neuron_spike_times = np.multiply(train_spikes[:, ii],
                                                      train_spike_times)
                nest.SetStatus([ii_spike_gen],
                               {"spike_times": iter_neuron_spike_times[
                                   iter_neuron_spike_times != 0],
                                "spike_weights": [1500.] * len(
                                    iter_neuron_spike_times[
                                        iter_neuron_spike_times != 0])}
                               )

    def plot_all(self, idx, save=True):
        spikes = nest.GetStatus(self.input_spike_detector, keys="events")[0]
        visualize.spike_plot(spikes, "Input spikes",
                             gen_idx=self.gen_idx, idx=0, save=save)
        # visualize.plot_data(idx, self.mean_ca_e, self.mean_ca_i,
        #                     self.total_connections_e,
        #                     self.total_connections_i)
        visualize.plot_fr(idx=idx, mean_ca_e=self.mean_ca_e,
                          mean_ca_i=self.mean_ca_i, save=save)
        # visualize.plot_output(idx, self.mean_ca_e_out)

    def create_individual(self, size_e, size_i):
        weights_e = np.random.normal(self.psc_e, 100., size_e)
        weights_i = np.random.normal(self.psc_i, 100., size_i)
        return {'weights_e': weights_e, 'weights_i': weights_i}

    def simulate(self, traj):
        # get indices
        self.gen_idx = traj.individual.generation
        self.ind_idx = traj.individual.ind_idx
        # prepare the connections etc.
        self.prepare_network()
        train_px = traj.individual.train_px_one
        target = traj.individual.targets
        self.set_external_input(iteration=self.gen_idx, train_px_one=train_px,
                                target=target)
        self.weights_e = traj.individual.weights_e
        self.weights_i = traj.individual.weights_i
        self.replace_weights(self.gen_idx, self.ind_idx, self.weights_e,
                             self.parameters.path, typ='e')
        self.replace_weights(self.gen_idx, self.ind_idx, self.weights_i,
                             self.parameters.path, typ='i')
        # Warm up simulation
        print("Starting simulation")
        if self.gen_idx < 1:
            print('Warm up')
            nest.Simulate(self.warm_up_time)
            print('Warm up done')
        if self.parameters.record_spiking_firingrate:
            self.clear_spiking_events()
        # start simulation
        sim_steps = np.arange(0, self.t_sim, self.record_interval)
        for j, step in enumerate(sim_steps):
            nest.Simulate(self.record_interval)
            if j % 20 == 0:
                print("Progress: " + str(j / 2) + "%")
            if self.parameters.record_spiking_firingrate:
                self.record_fr(indx=j, gen_idx=self.gen_idx,
                               save=self.parameters.save_plot)
                self.clear_spiking_events()
            else:
                self.record_ca()
            self.record_connectivity()
        print("Simulation loop finished successfully")
        model_out = softmax(
            [self.mean_ca_e[j][-1] for j in range(10)])
        label = self.target_labels[-1]
        target = np.zeros(10)
        target[label] = 1.0
        fitness = ((target - model_out) ** 2).sum()
        print(fitness)
        return dict(fitness=fitness, model_out=model_out)

    @staticmethod
    def replace_weights(gen_idx, ind_idx, weights, path='.', typ='e'):
        # Read the connections, i.e. sources and targets
        conns = pd.read_csv(
            os.path.join(path, '{}_connections_g{}_i{}.csv'.format(typ,
                                                                   gen_idx,
                                                                   ind_idx)))
        # weights = traj.individual.connection_weights

        sources = conns['source'].values
        targets = conns['target'].values
        # weights = conns['weight'].values
        print('now replacing connection weights')
        for (s, t, w) in zip(sources, targets, weights):
            syn_spec = {'weight': w}
            nest.Connect(tuple([s]), tuple([t]), syn_spec=syn_spec,
                         conn_spec='one_to_one')

    @staticmethod
    def save_connections(conn, gen_idx, ind_idx, path='.', typ='e'):
        status = nest.GetStatus(conn)
        d = OrderedDict({'source': [], 'target': []})
        for elem in status:
            d['source'].append(elem.get('source'))
            d['target'].append(elem.get('target'))
            # d['weight'].append(elem.get('weight'))
        df = pd.DataFrame(d)
        df.to_pickle(
            os.path.join(path,
                         '{}_connections_g{}_i{}.pkl'.format(typ, gen_idx,
                                                             ind_idx)))
        df.to_csv(
            os.path.join(path,
                         '{}_connections_g{}_i{}.csv'.format(typ, gen_idx,
                                                             ind_idx)))

    def checkpoint(self, ids):
        # Input connections
        connections = nest.GetStatus(nest.GetConnections(self.nodes_in))
        f = open('conn_input_{}.bin'.format(ids), "wb")
        pickle.dump(connections, f, pickle.HIGHEST_PROTOCOL)
        f.close()

        # Bulk connections
        connections = nest.GetStatus(nest.GetConnections(self.nodes_e))
        f = open('conn_bulke_{}.bin'.format(ids), "wb")
        pickle.dump(connections, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        connections = nest.GetStatus(nest.GetConnections(self.nodes_i))
        f = open('conn_bulki_{}.bin'.format(ids), "wb")
        pickle.dump(connections, f, pickle.HIGHEST_PROTOCOL)
        f.close()

        # # Out connections
        # connections = nest.GetStatus(nest.GetConnections(self.nodes_out_e[0]))
        # f = open('conn_oute_0_{}.bin'.format(ids), "wb")
        # pickle.dump(connections, f, pickle.HIGHEST_PROTOCOL)
        # f.close()
        # connections = nest.GetStatus(nest.GetConnections(self.nodes_out_i[0]))
        # f = open('conn_outi_0_{}.bin'.format(ids), "wb")
        # pickle.dump(connections, f, pickle.HIGHEST_PROTOCOL)
        # f.close()


def remove_files(extensions):
    files = []
    for ext in extensions:
        files.extend(glob.glob(ext))
        print(files)
        try:
            for f in files:
                os.remove(f)
        except OSError as ose:
            print(ose)


if __name__ == "__main__":
    target_label = ['3', '0']
    other_label = ['2', '1', '4', '5', '6', '7', '8', '9']
    adapt_neuron = AdaptiveOptimizee()
    data_set = adapt_neuron.get_mnist_data(target_label, other_label)
    adapt_neuron.set_mnist_data(data_set)
    remove_files(['*.eps', '*.spikes'])
    # for saving the firing rates
    fr_dict = {}
    for i in range(1000):
        adapt_neuron.gen_idx = i
        adapt_neuron.simulate(iteration=i, target_label=target_label,
                              other_label=other_label,
                              record_spiking_fr=True,
                              save_plot=False)
        adapt_neuron.plot_all(idx=i, save=False)
        key = adapt_neuron.target_labels[-1]
        if key not in fr_dict:
            fr_dict[key] = {"ex": [],
                            "inh": []}
            fr_dict[key]["ex"].append(adapt_neuron.mean_ca_e)
            fr_dict[key]["inh"].append(adapt_neuron.mean_ca_i)

        else:
            fr_dict[key]["ex"].append(adapt_neuron.mean_ca_e)
            fr_dict[key]["inh"].append(adapt_neuron.mean_ca_i)
        adapt_neuron.clear_records()
    # save firing rates
    np.save('firing_rates.npy', fr_dict)
