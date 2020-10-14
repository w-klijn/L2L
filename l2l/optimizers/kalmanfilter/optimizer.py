import glob
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn.preprocessing as pp

from collections import namedtuple
from l2l.optimizers.kalmanfilter.enkf import EnsembleKalmanFilter as EnKF
from l2l import dict_to_list
from l2l.optimizers.optimizer import Optimizer
from l2l.optimizers.crossentropy.distribution import Gaussian

import l2l.optimizers.kalmanfilter.data as data

logger = logging.getLogger("optimizers.kalmanfilter")

EnsembleKalmanFilterParameters = namedtuple(
    'EnsembleKalmanFilter', ['gamma', 'maxit', 'n_iteration',
                             'pop_size', 'n_batches', 'online', 'seed', 'path',
                             'stop_criterion']
)

EnsembleKalmanFilterParameters.__doc__ = """
:param gamma: float, A small value, multiplied with the eye matrix  
:param maxit: int, Epochs to run inside the Kalman Filter
:param n_iteration: int, Number of iterations to perform
:param pop_size: int, Minimal number of individuals per simulation.
    Corresponds to number of ensembles
:param n_batches: int, Number of mini-batches to use for training
:param online: bool, Indicates if only one data point will used, 
    Default: False
:param sampling_generation: After `sampling_generation` steps a gaussian sampling 
    on the parameters of the best individual is done, ranked by the fitness
    value 
:param seed: The random seed used to sample and fit the distribution. 
    Uses a random generator seeded with this seed.
:param path: String, Root path for the file saving and loading the connections
:param stop_criterion: float, When the current fitness is smaller or equal the 
    `stop_criterion` the optimization in the outer loop ends
"""


class EnsembleKalmanFilter(Optimizer):
    """
    Class for an Ensemble Kalman Filter optimizer
    """

    def __init__(self, traj,
                 optimizee_prepare,
                 optimizee_create_individual,
                 optimizee_fitness_weights,
                 parameters,
                 optimizee_bounding_func=None):
        super().__init__(traj,
                         optimizee_create_individual=optimizee_create_individual,
                         optimizee_fitness_weights=optimizee_fitness_weights,
                         parameters=parameters,
                         optimizee_bounding_func=optimizee_bounding_func)

        self.optimizee_bounding_func = optimizee_bounding_func
        self.optimizee_create_individual = optimizee_create_individual
        self.optimizee_fitness_weights = optimizee_fitness_weights
        self.optimizee_prepare = optimizee_prepare
        self.parameters = parameters

        traj.f_add_parameter('gamma', parameters.gamma, comment='Noise level')
        traj.f_add_parameter('maxit', parameters.maxit,
                             comment='Maximum iterations')
        traj.f_add_parameter('n_iteration', parameters.n_iteration,
                             comment='Number of iterations to run')
        traj.f_add_parameter('n_batches', parameters.n_batches)
        traj.f_add_parameter('online', parameters.online)
        # TODO reactivate?
        # traj.f_add_parameter('sampling_generation',
        #                      parameters.sampling_generation)
        traj.f_add_parameter('seed', np.uint32(parameters.seed),
                             comment='Seed used for random number generation '
                                     'in optimizer')
        traj.f_add_parameter('pop_size', parameters.pop_size)
        traj.f_add_parameter('path', parameters.path,
                             comment='Root folder for the simulation')
        traj.f_add_parameter('stop_criterion', parameters.stop_criterion,
                             comment='stopping threshold')
        #: The population (i.e. list of individuals) to be evaluated at the
        # next iteration
        size_e, size_i = self.optimizee_prepare()  # Change 0 for a run id?
        _, self.optimizee_individual_dict_spec = dict_to_list(
            self.optimizee_create_individual(size_e, size_i),
            get_dict_spec=True)

        traj.results.f_add_result_group('generation_params')

        # Set the random state seed for distribution
        self.random_state = np.random.RandomState(traj.parameters.seed)

        current_eval_pop = [self.optimizee_create_individual(size_e, size_i)
                            for _ in range(parameters.pop_size)]

        if optimizee_bounding_func is not None:
            current_eval_pop = [self.optimizee_bounding_func(ind) for ind in
                                current_eval_pop]

        self.eval_pop = current_eval_pop
        self.best_fitness = 0.
        self.best_individual = None
        self.current_fitness = np.inf

        # self.targets = parameters.observations

        # MNIST DATA HANDLING
        self.target_label = ['0', '1']
        self.other_label = ['2', '3', '4', '5', '6', '7', '8', '9']
        self.train_set = None
        self.train_labels = None
        self.other_set = None
        self.other_labels = None
        self.test_set = None
        self.test_labels = None
        self.test_set_other = None
        self.test_labels_other = None
        # get the targets
        self.get_mnist_data()
        if self.train_labels:
            self.optimizee_labels, self.random_ids = self.randomize_labels(
                self.train_labels, size=10)
        else:
            raise AttributeError('Train Labels are not set, please check.')

        for e in self.eval_pop:
            e["targets"] = self.optimizee_labels
            e["train_set"] = [self.train_set[r] for r in self.random_ids]
        self.g = 0

        self._expand_trajectory(traj)

    def get_mnist_data(self):
        self.train_set, self.train_labels, self.test_set, self.test_labels = \
            data.fetch(path='./mnist784_dat/',
                       labels=self.target_label)
        self.other_set, self.other_labels, self.test_set_other, self.test_labels_other = data.fetch(
            path='./mnist784_dat/', labels=self.other_label)

    @staticmethod
    def randomize_labels(labels, size):
        """
        Randomizes given labels `labels` with size `size`.

        :param labels: list of strings with labels
        :param size: int, size of how many labels should be returned
        :return list of randomized labels
        :return list of random numbers used to randomize the `labels` list
        """
        rnd = np.random.randint(low=0, high=len(labels), size=size)
        return [int(labels[i]) for i in rnd], rnd

    def post_process(self, traj, fitnesses_results):
        self.eval_pop.clear()

        individuals = traj.individuals[traj.generation]
        gamma = np.eye(len(self.target_label)) * traj.gamma

        ensemble_size = traj.pop_size
        # TODO before scaling the weights, check for the shapes and adjust
        #  with `_sample_from_individual`
        # weights = [traj.current_results[i][1]['connection_weights'] for i in
        #           range(ensemble_size)]
        weights = [np.concatenate(
            (individuals[i].weights_e, individuals[i].weights_i))
            for i in range(ensemble_size)]
        fitness = [traj.current_results[i][1]['fitness'] for i in
                   range(ensemble_size)]
        self.current_fitness = np.min(fitness)

        # TODO make sampling optional
        # weights = self._sample_from_individual(weights, fitness, bins=10000)
        # ens, scaler = self._scale_weights(weights, normalize=True,
        #                                   method=pp.MinMaxScaler)
        ens = np.array(weights)
        # ens = ens / np.max(ens)
        model_outs = np.array([traj.current_results[i][1]['model_out'] for i in
                               range(ensemble_size)])
        model_outs = model_outs.reshape((ensemble_size,
                                         len(self.target_label),
                                         traj.n_batches))
        logger.info('Sorted Fitness {}'.format(np.sort([traj.current_results[
                                                            i][1]['fitness']
                                                        for i in
                                                        range(ensemble_size)])))

        enkf = EnKF(maxit=traj.maxit,
                    online=traj.online,
                    n_batches=traj.n_batches)
        enkf.fit(ensemble=ens,
                 ensemble_size=ensemble_size,
                 observations=np.array(self.optimizee_labels),
                 model_output=model_outs,
                 gamma=gamma)
        # These are all the updated weights for each ensemble
        results = enkf.ensemble.cpu().numpy()  # scaler.inverse_transform(enkf.ensemble)
        self.plot_distribution(weights=results, gen=traj.generation, mean=True)

        generation_name = 'generation_{}'.format(traj.generation)
        traj.results.generation_params.f_add_result_group(generation_name)

        generation_result_dict = {
            'generation': traj.generation,
            'connection_weights': results
        }
        traj.results.generation_params.f_add_result(
            generation_name + '.algorithm_params', generation_result_dict)

        # Produce the new generation of individuals
        if self.g < len(self.train_labels) or \
                traj.stop_criterion <= self.current_fitness \
                or self.g < traj.n_iteration:
            # Create new individual based on the results of the update from the EnKF.
            new_individual_list = [
                {'weights_e': results[i][:len(individuals[i].weights_e)],
                 'weights_i': results[i][len(individuals[i].weights_i):],
                 'train_set': self.train_set,
                 'targets': self.optimizee_labels} for i in
                range(ensemble_size)]

            # Check this bounding function
            if self.optimizee_bounding_func is not None:
                new_individual_list = [self.optimizee_bounding_func(ind) for
                                       ind in new_individual_list]

            fitnesses_results.clear()
            self.eval_pop = new_individual_list
            self.g += 1  # Update generation counter
            self._expand_trajectory(traj)

    @staticmethod
    def _scale_weights(weights, normalize=False, method=pp.MinMaxScaler,
                       **kwargs):
        scaler = 0.
        if normalize:
            scaler = method(**kwargs)
            weights = scaler.fit_transform(weights)
        return weights, scaler

    @staticmethod
    def _sample_from_individual(individuals, fitness, bins='auto',
                                method='interpolate'):
        """
        The lengths of the individuals may differ. To fill the individuals to
        the same length sample values from the distribution of the individual
        with the best fitness.
        """
        # check if the sizes are different otherwise skip
        if len(set(
                [len(individuals[i]) for i in range(len(individuals))])) == 1:
            return individuals
        # best fitness should be here ~ 0 (which means correct choice)
        idx = np.argmin(fitness)
        best_ind = individuals[idx]
        best_min = np.min(best_ind)
        best_max = np.max(best_ind)
        hist = np.histogram(best_ind, bins)
        if method == "interpolate":
            # for interpolation it is better to reduce the max and min values
            best_max -= 100
            best_min += 100
            hist_dist = scipy.interpolate.interp1d(hist[1][:-1], hist[0])
        elif method == 'rv_histogram':
            hist_dist = scipy.stats.rv_histogram(hist)
        else:
            raise KeyError('Sampling method {} not known'.format(method))
        # get the longest individual
        longest_ind = individuals[np.argmax([len(ind) for ind in individuals])]
        new_inds = []
        for inds in individuals:
            subs = len(longest_ind) - len(inds)
            if subs > 0:
                rnd = np.random.uniform(best_min, best_max, subs)
                inds.extend(hist_dist(rnd))
                new_inds.append(inds)
            else:
                # only if subs is the longest individual
                new_inds.append(inds)
        return new_inds

    @staticmethod
    def plot_distribution(weights, gen, mean=True):
        """ Plots the weights as a histogram """
        if mean:
            plt.hist(weights.mean(0))
        else:
            plt.hist(weights)
        plt.savefig('weight_distributions_gen{}.eps'.format(gen), format='eps')
        plt.close()

    @staticmethod
    def _create_individual_distribution(random_state, weights,
                                        ensemble_size):
        dist = Gaussian()
        dist.init_random_state(random_state)
        dist.fit(weights)
        new_individuals = dist.sample(ensemble_size)
        return new_individuals

    def _new_individuals(self, traj, fitnesses, individuals, ensemble_size):
        """
        Sample new individuals by first ranking and then sampling from a
        gaussian distribution.
        """
        ranking_idx = list(reversed(np.argsort(fitnesses)))
        best_fitness = fitnesses[ranking_idx][0]
        best_ranking_idx = ranking_idx[0]
        best_individual = individuals[best_ranking_idx]
        # now do the sampling
        params = [
            self._create_individual_distribution(self.random_state,
                                                 individuals[
                                                     best_ranking_idx].params,
                                                 ensemble_size)
            for _ in range(traj.pop_size)]
        return params, best_fitness, best_individual

    @staticmethod
    def _remove_files(suffixes):
        for suffix in suffixes:
            files = glob.glob('*.{}'.format(suffix))
            try:
                [os.remove(fl) for fl in files]
            except OSError as ose:
                print('Error {} {}'.format(files, ose))

    def end(self, traj):
        """
        Run any code required to clean-up, print final individuals etc.
        :param  ~l2l.utils.trajectory.Trajectory traj: The  trajectory that contains the parameters and the
            individual that we want to simulate. The individual is accessible using `traj.individual` and parameter e.g.
            param1 is accessible using `traj.param1`
        """
        traj.f_add_result('final_individual', self.best_individual)

        logger.info(
            "The last individual {} was with fitness {}".format(
                self.best_individual, self.best_fitness))
        logger.info("-- End of (successful) EnKF optimization --")
