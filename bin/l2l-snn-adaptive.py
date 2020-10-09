from l2l.utils.experiment import Experiment
from l2l.optimizees.snn.adaptive_optimizee import AdaptiveOptimizee, \
    AdaptiveOptimizeeParameters
from l2l.optimizers.kalmanfilter import EnsembleKalmanFilter, \
    EnsembleKalmanFilterParameters


def run_experiment():
    experiment = Experiment(root_dir_path='../results')
    traj, _ = experiment.prepare_experiment(jube_parameter={}, name="L2L-ENKF")

    # Optimizee params
    optimizee_seed = 123
    optimizee_parameters = AdaptiveOptimizeeParameters(
        seed=optimizee_seed, path=experiment.root_dir_path,
        record_spiking_firingrate=True, save_plot=False)
    # Inner-loop simulator
    optimizee = AdaptiveOptimizee(traj, optimizee_parameters)

    # Outer-loop optimizer initialization
    optimizer_seed = 1234
    pop_size = 2
    optimizer_parameters = EnsembleKalmanFilterParameters(gamma=0.01,
                                                          maxit=1,
                                                          n_ensembles=pop_size,
                                                          n_iteration=3,
                                                          pop_size=pop_size,
                                                          n_batches=1,
                                                          online=False,
                                                          seed=1234,
                                                          stop_criterion=1e-2,
                                                          path=experiment.root_dir_path)

    optimizer = EnsembleKalmanFilter(traj,
                                     optimizee_prepare=optimizee.connect_network,
                                     optimizee_create_individual=optimizee.create_individual,
                                     optimizee_fitness_weights=(1.,),
                                     parameters=optimizer_parameters,
                                     optimizee_bounding_func=None)
    experiment.run_experiment(optimizee, optimizee_parameters, optimizer,
                              optimizer_parameters)

    experiment.end_experiment(optimizer)


def main():
    run_experiment()


if __name__ == '__main__':
    main()
