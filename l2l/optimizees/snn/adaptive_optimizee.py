from collections import OrderedDict, namedtuple
from itertools import permutations, product
from l2l.optimizees.optimizee import Optimizee
from l2l.optimizees.snn import spike_generator, visualize
from scipy.special import softmax

import json
import glob
import numpy as np
import nest
import os
import pandas as pd
import pathlib
import pickle

AdaptiveOptimizeeParameters = namedtuple(
    'AdaptiveOptimizeeParameters', ['path',
                                    'record_spiking_firingrate',
                                    'save_plot'])


class AdaptiveOptimizee(Optimizee):
    def __init__(self, traj, parameters):
        super().__init__(traj)
        self.parameters = parameters
        fp = pathlib.Path(__file__).parent.absolute()
        print(os.path.join(fp, 'config.json'))
        with open(
                os.path.join(fp, 'config.json')) as jsonfile:
            self.config = json.load(jsonfile)
        seed = np.uint32(self.config['seed'])
        self.random_state = np.random.RandomState(seed=seed)
        nest.SetKernelStatus({"rng_seeds": [seed]})
        self.t_sim = self.config['t_sim']
        self.input_type = self.config['input_type']
        # Resolution, simulation steps in [ms]
        self.dt = self.config['dt']
        self.neuron_model = self.config['neuron_model']
        # Indices per generation and individual
        self.gen_idx = traj.individual.generation
        self.ind_idx = traj.individual.ind_idx

        # Number of neurons per layer
        self.n_input_neurons = self.config['n_input']
        self.n_bulk_ex_neurons = self.config['n_bulk_ex']
        self.n_bulk_in_neurons = self.config['n_bulk_in']
        self.n_neurons_out_e = self.config['n_out_ex']
        self.n_neurons_out_i = self.config['n_out_in']
        self.n_output_clusters = self.config['n_output_clusters']
        self.psc_e = self.config['psc_e']
        self.psc_i = self.config['psc_i']
        self.psc_ext = self.config['psc_ext']
        self.bg_rate = self.config['bg_rate']
        self.record_interval = self.config['record_interval']
        self.warm_up_time = self.config['warm_up_time']
        self.cooling_time = self.config['cooling_time']

        # Init of nodes
        self.nodes_in = None
        self.nodes_e = None
        self.nodes_i = None
        self.nodes_out_e = []
        self.nodes_out_i = []
        # Init of generators and noise
        self.pixel_rate_generators = None
        self.noise = None
        # Init of spike detectors
        self.input_spike_detector = None
        self.bulks_detector_ex = None
        self.bulks_detector_in = None
        self.out_detector_e = None
        self.out_detector_i = None
        self.rates = None
        self.target_px = None
        # Lists for connections
        self.total_connections_e = []
        self.total_connections_i = []
        self.total_connections_out_e = []
        self.total_connections_out_i = []
        # Lists for firing rates
        self.mean_ca_e = []
        self.mean_ca_i = []
        self.mean_ca_out_e = [[] for _ in range(self.n_output_clusters)]
        self.mean_ca_out_i = [[] for _ in range(self.n_output_clusters)]
        # Lists for labels
        self.target_labels = []
        self.random_ids = []

    def connect_network(self):
        self.prepare_network()
        # Do the connections
        self.connect_internal_bulk()
        self.connect_external_input()
        self.connect_spike_detectors()
        self.connect_noise_bulk()
        self.connect_internal_out()
        self.connect_bulk_to_out()
        self.connect_noise_out()
        # create connection types of
        # {'ee', 'ei', 'eoeo', 'eoio', 'ie', 'ii', 'ioeo', 'ioio'}
        # e/i is the connection type, o indicates output
        # eo means excitatory ouput connections
        perms = set([''.join(p) for p in permutations('eeii', 2)])
        out_perms = set([''.join(p) for p in product(['eo', 'io'], repeat=2)])
        # only eeo, eio, ieo and iio are going to be optimized though
        # set([''.join(p) product(['e', 'i'], ['eo', 'io'], repeat=1)])
        optim_perms = {'eeo', 'eio', 'ieo', 'iio'}
        perms = perms.union(out_perms, optim_perms)
        # need dictionary to store the length specific to the outputs
        d = {}
        for p in perms:
            source, target = self._get_net_structure(p)
            conns = nest.GetConnections(
                source=tuple(np.ravel(source)), target=tuple(np.ravel(target)))
            self.save_connections(conns,
                                  path=self.parameters.path,
                                  typ=p)
            d[p] = len(conns)
        return d['eeo'], d['eio'], d['ieo'], d['iio']

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

    def _get_net_structure(self, typ):
        # types: {'ee', 'ei', 'eoeo', 'eoio', 'ie', 'ii', 'ioeo', 'ioio'}
        if typ == 'ee':
            return self.nodes_e, self.nodes_e
        elif typ == 'ei':
            return self.nodes_e, self.nodes_i
        elif typ == 'ie':
            return self.nodes_i, self.nodes_e
        elif typ == 'ii':
            return self.nodes_i, self.nodes_i
        elif typ == 'eeo':
            return self.nodes_e, self.nodes_out_e
        elif typ == 'eio':
            return self.nodes_e, self.nodes_out_i
        elif typ == 'ieo':
            return self.nodes_i, self.nodes_out_e
        elif typ == 'iio':
            return self.nodes_i, self.nodes_out_i
        elif typ == 'eoeo':
            return self.nodes_out_e, self.nodes_out_e
        elif typ == 'eoio':
            return self.nodes_out_e, self.nodes_out_i
        elif typ == 'ioeo':
            return self.nodes_out_i, self.nodes_out_e
        elif typ == 'ioio':
            return self.nodes_out_i, self.nodes_out_i

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
        for i in range(self.n_output_clusters):
            self.nodes_out_e.append(nest.Create(
                self.neuron_model, self.n_neurons_out_e))
            self.nodes_out_i.append(nest.Create(
                self.neuron_model, self.n_neurons_out_i))
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
            self.out_detector_e = nest.Create("spike_detector",
                                              self.n_output_clusters,
                                              params={"withgid": True,
                                                      "withtime": True,
                                                      "to_file": True,
                                                      "label": "out_e",
                                                      "file_extension": "spikes"})
            self.out_detector_i = nest.Create("spike_detector",
                                              self.n_output_clusters,
                                              params={"withgid": True,
                                                      "withtime": True,
                                                      "to_file": True,
                                                      "label": "out_i",
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
            self.pixel_rate_generators = nest.Create(
                "poisson_generator", len(rates))

    def connect_spike_detectors(self):
        # Input
        nest.Connect(self.nodes_in, self.input_spike_detector)
        # BULK
        nest.Connect(self.nodes_e, self.bulks_detector_ex)
        nest.Connect(self.nodes_i, self.bulks_detector_in)
        # Out
        for j in range(self.n_output_clusters):
            nest.Connect(self.nodes_out_e[j], [self.out_detector_e[j]])
            nest.Connect(self.nodes_out_i[j], [self.out_detector_i[j]])

    def connect_noise_bulk(self):
        poisson_gen = nest.Create("poisson_generator", 1, {'rate': 10000.0}, )
        syn_dict = {"model": "static_synapse", "weight": 1}
        syn_dict_i = {"model": "static_synapse", "weight": 1}
        nest.Connect(poisson_gen, self.nodes_e, "all_to_all",
                     syn_spec=syn_dict)
        nest.Connect(poisson_gen, self.nodes_i, "all_to_all",
                     syn_spec=syn_dict_i)

    def connect_noise_out(self):
        poisson_gen = nest.Create("poisson_generator", 1, {'rate': 10000.0}, )
        syn_dict = {"model": "static_synapse", "weight": 1}
        syn_dict_i = {"model": "static_synapse", "weight": 1}
        for j in range(self.n_output_clusters):
            nest.Connect(poisson_gen, self.nodes_out_e[j], "all_to_all",
                         syn_spec=syn_dict)
            nest.Connect(poisson_gen, self.nodes_out_i[j], "all_to_all",
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

    def connect_internal_out(self):
        # Connect out
        conn_dict = {'rule': 'fixed_indegree', 'indegree': 2}
        syn_dict = {"model": "random_synapse"}
        conn_dict_i = {'rule': 'fixed_indegree', 'indegree': 2}
        syn_dict_i = {"model": "random_synapse"}
        for ii in range(self.n_output_clusters):
            nest.Connect(self.nodes_out_e[ii], self.nodes_out_e[ii], conn_dict,
                         syn_spec=syn_dict)
            nest.Connect(self.nodes_out_e[ii], self.nodes_out_i[ii], conn_dict,
                         syn_spec=syn_dict)
            nest.Connect(self.nodes_out_i[ii], self.nodes_out_e[ii],
                         conn_dict_i, syn_spec=syn_dict_i)
            nest.Connect(self.nodes_out_i[ii], self.nodes_out_i[ii],
                         conn_dict_i, syn_spec=syn_dict_i)

    def connect_bulk_to_out(self):
        # Bulk to out
        conn_dict_e = {'rule': 'fixed_indegree',
                       # 0.3 * self.number_out_exc_neurons
                       'indegree': int(0.15 * self.n_bulk_ex_neurons)}
        conn_dict_i = {'rule': 'fixed_indegree',
                       # 0.2 * self.number_out_exc_neurons
                       'indegree': int(0.1 * self.n_bulk_in_neurons)}
        syn_dict_e = {"model": "random_synapse",
                      'weight': {"distribution": "normal",
                                 "mu": self.psc_e,
                                 "sigma": 100.}}
        syn_dict_i = {"model": "random_synapse_i",
                      'weight': {"distribution": "normal",
                                 "mu": self.psc_i,
                                 "sigma": 100.}}
        for j in range(self.n_output_clusters):
            nest.Connect(self.nodes_e, self.nodes_out_e[j], conn_dict_e,
                         syn_spec=syn_dict_e)
            nest.Connect(self.nodes_e, self.nodes_out_i[j], conn_dict_i,
                         syn_spec=syn_dict_i)
            nest.Connect(self.nodes_i, self.nodes_out_e[j], conn_dict_i,
                         syn_spec=syn_dict_i)
            nest.Connect(self.nodes_i, self.nodes_out_i[j], conn_dict_e,
                         syn_spec=syn_dict_e)

    def connect_external_input(self):
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
        for i in range(self.n_output_clusters):
            nest.SetStatus([self.out_detector_e[i]], "n_events", 0)
            nest.SetStatus([self.out_detector_i[i]], "n_events", 0)

    def record_fr(self, indx, gen_idx, record_out=False, save=True):
        """ Records firing rates """
        n_recorded_bulk_ex = self.n_bulk_ex_neurons
        n_recorded_bulk_in = self.n_bulk_in_neurons
        self.mean_ca_e.append(
            nest.GetStatus(self.bulks_detector_ex, "n_events")[
                0] * 1000.0 / (self.record_interval * n_recorded_bulk_ex))
        self.mean_ca_i.append(
            nest.GetStatus(self.bulks_detector_in, "n_events")[
                0] * 1000.0 / (self.record_interval * n_recorded_bulk_in))
        if record_out:
            for i in range(self.n_output_clusters):
                self.mean_ca_out_e[i].append(
                    nest.GetStatus([self.out_detector_e[i]], "n_events")[
                        0] * 1000.0 / (
                        self.record_interval * self.n_neurons_out_e))
                self.mean_ca_out_i[i].append(
                    nest.GetStatus([self.out_detector_i[i]], "n_events")[
                        0] * 1000.0 / (
                        self.record_interval * self.n_neurons_out_i))
        spikes = nest.GetStatus(self.bulks_detector_ex, keys="events")[0]
        visualize.spike_plot(spikes, "Bulk spikes",
                             idx=indx, gen_idx=gen_idx, save=save)
        spikes = nest.GetStatus(self.bulks_detector_ex, keys="events")[0]
        visualize.spike_plot(spikes, "Bulk spikes",
                             idx=indx, gen_idx=gen_idx, save=save)

    def record_ca(self, record_out=False):
        ca_e = nest.GetStatus(self.nodes_e, 'Ca'),  # Calcium concentration
        self.mean_ca_e.append(np.mean(ca_e))
        ca_i = nest.GetStatus(self.nodes_i, 'Ca'),  # Calcium concentration
        self.mean_ca_i.append(np.mean(ca_i))
        if record_out:
            for ii in range(self.n_output_clusters):
                # Calcium concentration
                ca_e = nest.GetStatus(self.nodes_out_e[ii], 'Ca'),
                self.mean_ca_out_e[ii].append(np.mean(ca_e))
                ca_i = nest.GetStatus(self.nodes_out_i[ii], 'Ca'),
                self.mean_ca_out_i[ii].append(np.mean(ca_i))

    def clear_records(self):
        self.mean_ca_i = []
        self.mean_ca_e = []
        self.mean_ca_out_e = [[] for _ in range(self.n_output_clusters)]
        self.mean_ca_out_i = [[] for _ in range(self.n_output_clusters)]
        nest.SetStatus(self.input_spike_detector, {"n_events": 0})
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

    def set_external_input(self, iteration, train_data, target):
        path = self.parameters.path
        save = self.parameters.save_plot
        # Save image for reference
        if save:
            visualize.plot_image(image=train_data, random_id=target,
                                 iteration=iteration, path=path, save=save)
        if self.input_type == 'greyvalue':
            rates = spike_generator.greyvalue(train_data,
                                              min_rate=1, max_rate=100)
            generator_stats = [{'rate': w} for w in rates]
            nest.SetStatus(self.pixel_rate_generators, generator_stats)
        elif self.input_type == 'greyvalue_sequential':
            rates = spike_generator.greyvalue_sequential(train_data,
                                                         min_rate=1,
                                                         max_rate=100,
                                                         start_time=0,
                                                         end_time=783)
            generator_stats = [{'rate': w} for w in rates]
            nest.SetStatus(self.pixel_rate_generators, generator_stats)
        else:
            train_spikes, train_spike_times = spike_generator.bellec_spikes(
                train_data, self.n_input_neurons, self.dt)
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
        if save:
            visualize.spike_plot(spikes, "Input spikes",
                                 gen_idx=self.gen_idx, idx=0, save=save)
            # visualize.plot_data(idx, self.mean_ca_e, self.mean_ca_i,
            #                     self.total_connections_e,
            #                     self.total_connections_i)
            visualize.plot_fr(idx=idx, mean_ca_e=self.mean_ca_e,
                              mean_ca_i=self.mean_ca_i, save=save)
            # visualize.plot_output(idx, self.mean_ca_e_out)

    def create_individual(self, size_eeo, size_eio, size_ieo, size_iio):
        mu = self.config['mu']
        sigma = self.config['sigma']
        weights_eeo = np.random.normal(mu, sigma, size_eeo)
        weights_eio = np.random.normal(mu, sigma, size_eio)
        weights_ieo = np.random.normal(mu, sigma, size_ieo)
        weights_iio = np.random.normal(mu, sigma, size_iio)
        return {'weights_eeo': weights_eeo, 'weights_eio': weights_eio,
                'weights_ieo': weights_ieo, 'weights_iio': weights_iio}

    def simulate(self, traj):
        # get indices
        self.gen_idx = traj.individual.generation
        self.ind_idx = traj.individual.ind_idx
        # prepare the connections etc.
        self.prepare_network()
        self.connect_external_input()
        self.connect_spike_detectors()
        train_set = traj.individual.train_set
        targets = traj.individual.targets
        weights_eeo = traj.individual.weights_eeo
        weights_eio = traj.individual.weights_eio
        weights_ieo = traj.individual.weights_ieo
        weights_iio = traj.individual.weights_iio
        self.replace_weights(weights_eeo,
                             self.parameters.path, typ='eeo')
        self.replace_weights(weights_eio,
                             self.parameters.path, typ='eio')
        self.replace_weights(weights_ieo,
                             self.parameters.path, typ='ieo')
        self.replace_weights(weights_iio,
                             self.parameters.path, typ='iio')
        # Warm up simulation
        print("Starting simulation")
        if self.gen_idx < 1:
            print('Warm up')
            nest.Simulate(self.warm_up_time)
            print('Warm up done')
        # start simulation
        fitnesses = []
        model_outs = []
        for idx, target in enumerate(targets):
            # cooling time, empty simulation
            print('Cooling period')
            # Clear input
            self.clear_input()
            nest.Simulate(self.cooling_time)
            print('Cooling done')
            self.clear_records()
            if self.parameters.record_spiking_firingrate:
                self.clear_spiking_events()
            self.set_external_input(iteration=self.gen_idx,
                                    train_data=train_set[idx],
                                    target=target)
            sim_steps = np.arange(0, self.t_sim, self.record_interval)
            for j, step in enumerate(sim_steps):
                # Do the simulation
                nest.Simulate(self.record_interval)
                if j % 20 == 0:
                    print("Progress: " + str(j / 2) + "%")
                if self.parameters.record_spiking_firingrate:
                    self.record_fr(indx=j, gen_idx=self.gen_idx,
                                   save=self.parameters.save_plot,
                                   record_out=True)
                    self.clear_spiking_events()
                else:
                    self.record_ca(record_out=True)
                self.record_connectivity()
            print("Simulation loop {} finished successfully".format(idx))
            print('Mean out e ', self.mean_ca_out_e)
            print('Mean e ', self.mean_ca_e)
            softm = softmax([self.mean_ca_out_e[j][-1] for j in
                             range(self.n_output_clusters)])
            argmax = np.argmax(softm)
            model_outs.append(softm)
            # one hot encoding
            label = np.eye(self.n_output_clusters)[target]
            pred = np.eye(self.n_output_clusters)[argmax]
            fitness = ((label - pred) ** 2).sum()
            fitnesses.append(fitness)
            print('Fitness {} for target {}, softmax {}, argmax {}'.format(
                fitness, target, softm, argmax))
            # clear lists
            self.clear_records()
        print('Fitness values {}, mean {}'.format(
            fitnesses, np.mean(fitnesses)))
        return dict(fitness=np.mean(fitnesses), model_out=model_outs)

    @staticmethod
    def replace_weights(weights, path='.', typ='e'):
        # Read the connections, i.e. sources and targets
        conns = pd.read_csv(
            os.path.join(path, '{}_connections.csv'.format(typ)))

        sources = conns['source'].values
        targets = conns['target'].values
        # weights = conns['weight'].values
        print('now replacing connection weights')
        for (s, t, w) in zip(sources, targets, weights):
            syn_spec = {'weight': w}
            nest.Connect(tuple([s]), tuple([t]), syn_spec=syn_spec,
                         conn_spec='one_to_one')

    @staticmethod
    def save_connections(conn, path='.', typ='e'):
        status = nest.GetStatus(conn)
        d = OrderedDict({'source': [], 'target': []})
        for elem in status:
            d['source'].append(elem.get('source'))
            d['target'].append(elem.get('target'))
            # d['weight'].append(elem.get('weight'))
        df = pd.DataFrame(d)
        df.to_pickle(
            os.path.join(path,
                         '{}_connections.pkl'.format(typ)))
        df.to_csv(
            os.path.join(path,
                         '{}_connections.csv'.format(typ)))

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
        connections = nest.GetStatus(nest.GetConnections(self.nodes_out_e[0]))
        f = open('conn_oute_0_{}.bin'.format(ids), "wb")
        pickle.dump(connections, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        connections = nest.GetStatus(nest.GetConnections(self.nodes_out_i[0]))
        f = open('conn_outi_0_{}.bin'.format(ids), "wb")
        pickle.dump(connections, f, pickle.HIGHEST_PROTOCOL)
        f.close()


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
