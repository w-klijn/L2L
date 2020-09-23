from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, minmax_scale, normalize
from sklearn.neural_network import MLPClassifier
from enum import Enum

Classifiers = Enum('MLP SVM PCA')


class Classifier:
    def __init__(self, clf, **kwargs):
        """
        Class to provide a classifier

        :param clf: Enum, possible enums are `MLP`, `SVM`, `PCA`
        :param kwargs: key word arguments for the classifiers
        """
        if clf == Classifiers.MLP:
            self.clf = MLPClassifier(solver='adam',
                                     alpha=1e-5,
                                     hidden_layer_sizes=kwargs.get(
                                         "hidden_layer_sizes", (500, 300)),
                                     max_iter=kwargs.get('max_iter', 300),
                                     random_state=kwargs.get("random_state",
                                                             1),
                                     **kwargs)
        elif self.clf == Classifiers.SVM:
            svm.SVC(gamma='scale', kernel='rbf', probability=True, **kwargs)
        elif self.clf == Classifiers.PCA:
            self.clf = PCA(n_components=kwargs.get("n_components", 2),
                           **kwargs)
        else:
            raise AttributeError('Enum {} not identified'.format(clf))
