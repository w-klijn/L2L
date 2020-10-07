from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, minmax_scale, normalize
from sklearn.neural_network import MLPClassifier
from enum import Enum

Classifiers = Enum('MLP SVM PCA')


class Classifier:
    def __init__(self, clf_name, **kwargs):
        """
        Class to provide a classifier

        :param clf_name: Enum, possible enums are `MLP`, `SVM`, `PCA`
        :param kwargs: key word arguments for the classifiers
        """
        self.clf_name = clf_name
        if clf_name == Classifiers.MLP:
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
            raise AttributeError('Enum {} not identified'.format(clf_name))

    def classify(self, x_train, y_train, x_test, y_test):
        self.clf.fit(x_train, y_train)
        score = self.clf.score(x_test, y_test)
        # get model output (only for mlp and svm)
        if self.clf_name == Classifiers.PCA:
            model_output = self.clf.get_precision()
        else:
            model_output = self.clf.predict_proba(x_train)
        return score, model_output
