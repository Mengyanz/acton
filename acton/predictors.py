"""Predictor classes."""

from abc import ABC, abstractmethod
import logging
from typing import Iterable, Sequence

import acton.database
import acton.kde_predictor
import GPy as gpy
import numpy
import sklearn.base
import sklearn.linear_model
import sklearn.model_selection
import sklearn.preprocessing
from numpy.random import multivariate_normal, gamma, multinomial


class Predictor(ABC):
    """Base class for predictors.

    Attributes
    ----------
    prediction_type : str
        What kind of predictions this class generates, e.g. classification.s
    """
    prediction_type = 'classification'

    @abstractmethod
    def fit(self, ids: Iterable[int]):
        """Fits the predictor to labelled data.

        Parameters
        ----------
        ids
            List of IDs of instances to train from.
        """

    @abstractmethod
    def predict(self, ids: Sequence[int]) -> (numpy.ndarray, numpy.ndarray):
        """Predicts labels of instances.

        Notes
        -----
            Unlike in scikit-learn, predictions are always real-valued.
            Predicted labels for a classification problem are represented by
            predicted probabilities of each class.

        Parameters
        ----------
        ids
            List of IDs of instances to predict labels for.

        Returns
        -------
        numpy.ndarray
            An N x T x C array of corresponding predictions.
        numpy.ndarray
            A N array of confidences (or None if not applicable).
        """

    @abstractmethod
    def reference_predict(
            self, ids: Sequence[int]) -> (numpy.ndarray, numpy.ndarray):
        """Predicts labels using the best possible method.

        Parameters
        ----------
        ids
            List of IDs of instances to predict labels for.

        Returns
        -------
        numpy.ndarray
            An N x 1 x C array of corresponding predictions.
        numpy.ndarray
            A N array of confidences (or None if not applicable).
        """


class _InstancePredictor(Predictor):
    """Wrapper for a scikit-learn instance.

    Attributes
    ----------
    _db : acton.database.Database
        Database storing features and labels.
    _instance : sklearn.base.BaseEstimator
        scikit-learn predictor instance.
    """

    def __init__(self, instance: sklearn.base.BaseEstimator,
                 db: acton.database.Database):
        """
        Arguments
        ---------
        instance
            scikit-learn predictor instance.
        db
            Database storing features and labels.
        """
        self._db = db
        self._instance = instance

    def fit(self, ids: Iterable[int]):
        """Fits the predictor to labelled data.

        Parameters
        ----------
        ids
            List of IDs of instances to train from.
        """
        features = self._db.read_features(ids)
        labels = self._db.read_labels([0], ids)
        self._instance.fit(features, labels.ravel())

    def predict(self, ids: Sequence[int]) -> (numpy.ndarray, None):
        """Predicts labels of instances.

        Notes
        -----
            Unlike in scikit-learn, predictions are always real-valued.
            Predicted labels for a classification problem are represented by
            predicted probabilities of each class.

        Parameters
        ----------
        ids
            List of IDs of instances to predict labels for.

        Returns
        -------
        numpy.ndarray
            An N x 1 x C array of corresponding predictions.
        numpy.ndarray
            A N array of confidences (or None if not applicable).
        """
        features = self._db.read_features(ids)
        try:
            probs = self._instance.predict_proba(features)
            return probs.reshape((probs.shape[0], 1, probs.shape[1])), None
        except AttributeError:
            probs = self._instance.predict(features)
            if len(probs.shape) == 1:
                return probs.reshape((probs.shape[0], 1, 1)), None
            else:
                raise NotImplementedError()

    def reference_predict(self, ids: Sequence[int]) -> (numpy.ndarray, None):
        """Predicts labels using the best possible method.

        Parameters
        ----------
        ids
            List of IDs of instances to predict labels for.

        Returns
        -------
        numpy.ndarray
            An N x 1 x C array of corresponding predictions.
        numpy.ndarray
            A N array of confidences (or None if not applicable).
        """
        return self.predict(ids)


def from_instance(predictor: sklearn.base.BaseEstimator,
                  db: acton.database.Database, regression: bool=False
                  ) -> Predictor:
    """Converts a scikit-learn predictor instance into a Predictor instance.

    Arguments
    ---------
    predictor
        scikit-learn predictor.
    db
        Database storing features and labels.
    regression
        Whether this predictor does regression (as opposed to classification).

    Returns
    -------
    Predictor
        Predictor instance wrapping the scikit-learn predictor.
    """
    ip = _InstancePredictor(predictor, db)
    if regression:
        ip.prediction_type = 'regression'
    return ip


def from_class(Predictor: type, regression: bool=False) -> type:
    """Converts a scikit-learn predictor class into a Predictor class.

    Arguments
    ---------
    Predictor
        scikit-learn predictor class.
    regression
        Whether this predictor does regression (as opposed to classification).

    Returns
    -------
    type
        Predictor class wrapping the scikit-learn class.
    """
    class Predictor_(_InstancePredictor):

        def __init__(self, db, **kwargs):
            super().__init__(instance=None, db=db)
            self._instance = Predictor(**kwargs)

    if regression:
        Predictor_.prediction_type = 'regression'

    return Predictor_


class Committee(Predictor):
    """A predictor using a committee of other predictors.

    Attributes
    ----------
    n_classifiers : int
        Number of logistic regression classifiers in the committee.
    subset_size : float
        Percentage of known labels to take subsets of to train the
        classifier. Lower numbers increase variety.
    _db : acton.database.Database
        Database storing features and labels.
    _committee : List[sklearn.linear_model.LogisticRegression]
        Underlying committee of logistic regression classifiers.
    _reference_predictor : Predictor
        Reference predictor trained on all known labels.
    """

    def __init__(self, Predictor: type, db: acton.database.Database,
                 n_classifiers: int=10, subset_size: float=0.6,
                 **kwargs: dict):
        """
        Parameters
        ----------
        Predictor
            Predictor to use in the committee.
        db
            Database storing features and labels.
        n_classifiers
            Number of logistic regression classifiers in the committee.
        subset_size
            Percentage of known labels to take subsets of to train the
            classifier. Lower numbers increase variety.
        kwargs
            Keyword arguments passed to the underlying Predictor.
        """
        self.n_classifiers = n_classifiers
        self.subset_size = subset_size
        self._db = db
        self._committee = [Predictor(db=db, **kwargs)
                           for _ in range(n_classifiers)]
        self._reference_predictor = Predictor(db=db, **kwargs)

    def fit(self, ids: Iterable[int]):
        """Fits the predictor to labelled data.

        Parameters
        ----------
        ids
            List of IDs of instances to train from.
        """
        # Get labels so we can stratify a split.
        labels = self._db.read_labels([0], ids)
        for classifier in self._committee:
            # Take a subsets to introduce variety.
            try:
                subset, _ = sklearn.model_selection.train_test_split(
                    ids, train_size=self.subset_size, stratify=labels)
            except ValueError:
                # Too few labels.
                subset = ids
            classifier.fit(subset)
        self._reference_predictor.fit(ids)

    def predict(self, ids: Sequence[int]) -> (numpy.ndarray, numpy.ndarray):
        """Predicts labels of instances.

        Notes
        -----
            Unlike in scikit-learn, predictions are always real-valued.
            Predicted labels for a classification problem are represented by
            predicted probabilities of each class.

        Parameters
        ----------
        ids
            List of IDs of instances to predict labels for.

        Returns
        -------
        numpy.ndarray
            An N x T x C array of corresponding predictions.
        numpy.ndarray
            A N array of confidences (or None if not applicable).
        """
        predictions = numpy.concatenate(
            [classifier.predict(ids)[0]
             for classifier in self._committee],
            axis=1)
        assert predictions.shape[:2] == (len(ids), len(self._committee))
        stdevs = predictions.std(axis=1).mean(axis=1)
        return predictions, stdevs

    def reference_predict(
            self, ids: Sequence[int]) -> (numpy.ndarray, numpy.ndarray):
        """Predicts labels using the best possible method.

        Parameters
        ----------
        ids
            List of IDs of instances to predict labels for.

        Returns
        -------
        numpy.ndarray
            An N x 1 x C array of corresponding predictions.
        numpy.ndarray
            A N array of confidences (or None if not applicable).
        """
        _, stdevs = self.predict(ids)
        return self._reference_predictor.predict(ids)[0], stdevs


def AveragePredictions(predictor: Predictor) -> Predictor:
    """Wrapper for a predictor that averages predicted probabilities.

    Notes
    -----
    This effectively reduces the number of predictors to 1.

    Arguments
    ---------
    predictor
        Predictor to wrap.

    Returns
    -------
    Predictor
        Predictor with averaged predictions.
    """
    predictor.predict_ = predictor.predict

    def predict(features: numpy.ndarray) -> (numpy.ndarray, numpy.ndarray):
        predictions, stdevs = predictor.predict_(features)
        predictions = predictions.mean(axis=1)
        return predictions.reshape(
            (predictions.shape[0], 1, predictions.shape[1])), stdevs

    predictor.predict = predict

    return predictor


class GPClassifier(Predictor):
    """Classifier using Gaussian processes.

    Attributes
    ----------
    max_iters : int
        Maximum optimisation iterations.
    label_encoder : sklearn.preprocessing.LabelEncoder
        Encodes labels as integers.
    model_ : gpy.models.GPClassification
        GP model.
    _db : acton.database.Database
        Database storing features and labels.
    """

    def __init__(self, db: acton.database.Database, max_iters: int=50000,
                 n_jobs: int=1):
        """
        Parameters
        ----------
        db
            Database.
        max_iters
            Maximum optimisation iterations.
        n_jobs
            Does nothing; here for compatibility with sklearn.
        """
        self._db = db
        self.max_iters = max_iters

    def fit(self, ids: Iterable[int]):
        """Fits the predictor to labelled data.

        Parameters
        ----------
        ids
            List of IDs of instances to train from.
        """
        features = self._db.read_features(ids)
        labels = self._db.read_labels([0], ids).ravel()
        self.label_encoder_ = sklearn.preprocessing.LabelEncoder()
        labels = self.label_encoder_.fit_transform(labels).reshape((-1, 1))
        if len(self.label_encoder_.classes_) > 2:
            raise ValueError(
                'GPClassifier only supports binary classification.')
        self.model_ = gpy.models.GPClassification(features, labels)
        self.model_.optimize('bfgs', max_iters=self.max_iters)

    def predict(self, ids: Sequence[int]) -> (numpy.ndarray, numpy.ndarray):
        """Predicts labels of instances.

        Notes
        -----
            Unlike in scikit-learn, predictions are always real-valued.
            Predicted labels for a classification problem are represented by
            predicted probabilities of each class.

        Parameters
        ----------
        ids
            List of IDs of instances to predict labels for.

        Returns
        -------
        numpy.ndarray
            An N x 1 x C array of corresponding predictions.
        numpy.ndarray
            A N array of confidences (or None if not applicable).
        """
        features = self._db.read_features(ids)
        p_predictions, variances = self.model_.predict(features)
        n_predictions = 1 - p_predictions
        predictions = numpy.concatenate([n_predictions, p_predictions], axis=1)

        logging.debug('Variance: {}'.format(variances))
        if isinstance(variances, float) and numpy.isnan(variances):
            variances = None
        else:
            variances = variances.ravel()
            assert variances.shape == (len(ids),)
        assert predictions.shape[1] == 2
        return predictions.reshape((-1, 1, 2)), variances

    def reference_predict(
            self, ids: Sequence[int]) -> (numpy.ndarray, numpy.ndarray):
        """Predicts labels using the best possible method.

        Parameters
        ----------
        ids
            List of IDs of instances to predict labels for.

        Returns
        -------
        numpy.ndarray
            An N x 1 x C array of corresponding predictions.
        numpy.ndarray
            A N array of confidences (or None if not applicable).
        """
        return self.predict(ids)


class TensorPredictor(Predictor):
    """Predict labels for each tensor entry.

    Attributes
    ----------
    _db : acton.database.Database
        Database storing features and labels.
    n_particles:
        Number of particles for Thompson sampling.
    n_relations:
        Number of relations (K)
    n_entities:
        Number of entities (N)
    n_dim
        Number of latent dimensions (D)
    var_r
        variance of prior of R
    var_e
        variance of prior of E
    var_x
        variance of X
    sample_prior
        indicates whether sample prior
    E
        P x N x D entity features
    R
        P x K x D x D relation features
    X
        K x N x N labels
    """

    def __init__(self,
                 db: acton.database.Database,
                 n_particles: int = 5,
                 var_r: float = 1.0, var_e: float = 1.0,
                 mean_r: float = 0.0, mean_e: float = 0.0,
                 var_x: float = 0.1,
                 sample_prior: bool = False,
                 n_jobs: int=1
                 ):
        """
        Arguments
        ---------
        db
            Database storing features and labels.
        n_particles:
            Number of particles for Thompson sampling.
        var_r
            variance of prior of R
        var_e
            variance of prior of E
        mean_r
            mean of prior of R
        mean_e
            mean of prior of E
        var_x
            variance of X
        sample_prior
            indicates whether sample prior
        """
        self._db = db
        self.n_particles = n_particles
        self.var_e = var_e
        self.var_r = var_r
        self.mean_e = mean_e
        self.mean_r = mean_r
        self.var_x = var_x

        self.p_weights = numpy.ones(self.n_particles) / self.n_particles

        self.sample_prior = sample_prior

        self.E, self.R = self._db.read_features()
        # X : numpy.ndarray
        #     Fully observed tensor with shape
        #     (n_relations, n_entities, n_entities)
        all_ = []
        self.X = self._db.read_labels(all_)  # read all labels

    def fit(self, ids: Iterable[tuple],
            n_initial_labels: int,
            inc_sub: bool,
            subn_entities: int,
            subn_relations: int,
            update_one: bool = False):
        """Update posteriors.

        Parameters
        ----------
        ids
            List of IDs of labelled instances.
        n_initial_labels
            number of initial labels
        inc_sub
            indicates whether increasing subsampling size when gets more labels
        subn_entities
            number of entities for subsampling
        subn_relations
            number of relations for subsampling
        update_one
            Boolean variable
            True: only update posterior entity and relations
                  related to the new label
            False: update all

        Returns
        -------
        seq : (numpy.ndarray, numpy.ndarray)
            Returns a updated posteriors for E and R.
        """
        # use certain number of subsampling, rather than percent
        # self.sub_percent = sub_percent
        # self.subn_entities = round(self.n_entities * self.sub_percent)
        # self.subn_relations = round(self.n_relations * self.sub_percent)

        assert self.n_particles == self.E.shape[0] == self.R.shape[0]
        self.n_relations = self.X.shape[0]
        self.n_entities = self.X.shape[1]
        self.n_dim = self.E.shape[2]
        assert self.E.shape[2] == self.R.shape[2]

        obs_mask = numpy.zeros_like(self.X)

        for _id in ids:
            r_k, e_i, e_j = _id
            obs_mask[r_k, e_i, e_j] = 1

        cur_obs = numpy.zeros_like(self.X)
        for k in range(self.n_relations):
            cur_obs[k][obs_mask[k] == 1] = self.X[k][obs_mask[k] == 1]

        # cur_obs[cur_obs.nonzero()] = 1
        self.obs_sum = numpy.sum(numpy.sum(obs_mask, 1), 1)
        self.valid_relations = \
            numpy.nonzero(numpy.sum(numpy.sum(self.X, 1), 1))[0]
        # totoal_size = self.n_relations * self.n_entities * self.n_dim

        # subsampling
        if numpy.sum(self.obs_sum) > 1000:
            self.subn_entities = 10
            self.subn_relations = 10
        else:
            self.subn_entities = int(subn_entities)
            self.subn_relations = int(subn_relations)

        self.features = numpy.zeros(
            [2 * self.n_entities * self.n_relations, self.n_dim])
        self.xi = numpy.zeros([2 * self.n_entities * self.n_relations])

        # only consider the situation where one element is recommended each time
        next_idx = ids[-1]

        self.p_weights *= \
            self.compute_particle_weight(next_idx, cur_obs, obs_mask)
        self.p_weights /= numpy.sum(self.p_weights)

        ESS = 1. / numpy.sum((self.p_weights ** 2))

        if ESS < self.n_particles / 2.:
            self.resample()

        if update_one:

            if n_initial_labels == len(ids):
                next_idxs = ids
            else:
                next_idxs = [next_idx]

            if isinstance(self.var_e, float) and isinstance(self.var_r, float)\
                    and isinstance(self.mean_e, float)\
                    and isinstance(self.mean_r, float):
                self.mean_e = numpy.ones((
                    self.n_particles, self.n_entities, self.n_dim)
                    ) * self.mean_e      # P x N x D
                self.mean_r = numpy.ones((
                    self.n_particles, self.n_relations, self.n_dim ** 2)
                    ) * self.mean_r    # P x K x D^2

                var_e = self.var_e
                var_r = self.var_r
                # P x N x D x D
                self.var_e = numpy.zeros((
                    self.n_particles, self.n_entities, self.n_dim, self.n_dim))
                # P x N x D^2 x D^2
                self.var_r = numpy.zeros((
                    self.n_particles, self.n_relations,
                    self.n_dim ** 2, self.n_dim ** 2))

                for p in range(self.n_particles):
                    for i in range(self.var_e.shape[1]):
                        self.var_e[p][i] = numpy.identity(self.n_dim) * var_e
                    for k in range(self.var_r.shape[1]):
                        self.var_r[p][k] = numpy.identity(
                            self.n_dim ** 2) * var_r

            for next_idx in next_idxs:
                for p in range(self.n_particles):

                    # self.mean_e[p], self.mean_r[p],\
                    # self.var_e[p], self.var_r[p]=\
                    self._sample_latent_variables(
                        next_idx, self.X[next_idx],
                        self.mean_e[p], self.mean_r[p],
                        self.var_e[p], self.var_r[p],
                        self.E[p], self.R[p]
                        )
            logging.debug('update one: {} using label: {}'.format(
                next_idx, self.X[next_idx]))
            # logging.debug('s_en_inv: {}'.format(s_en_inv))
            # logging.debug('ETR: {}'.format(ETR))
            # logging.debug('mean_r[0]: {}'.format(self.mean_r[0]))
            # logging.debug('E[0]: {}'.format(self.E[0]))

        else:
            if isinstance(self.var_e, float) and isinstance(self.var_r, float)\
                    and isinstance(self.mean_e, float)\
                    and isinstance(self.mean_r, float):
                self.var_e = list(numpy.ones(self.n_particles) * self.var_e)
                self.var_r = list(numpy.ones(self.n_particles) * self.var_r)
                self.mean_e = list(numpy.ones(self.n_particles) * self.mean_e)
                self.mean_r = list(numpy.ones(self.n_particles) * self.mean_r)
            if self.subn_entities == self.n_entities \
                    and self.subn_relations == self.n_relations:
                logging.debug("Sampling all.")
                sub_relids = None
                sub_entids = None
            else:
                logging.debug("Subsampling {} entities and {} relations".format(
                    self.subn_entities, self.subn_relations))
                sub_relids = numpy.random.choice(
                    self.n_relations, self.subn_relations, replace=False)
                sub_entids = numpy.random.choice(
                    self.n_entities, self.subn_entities, replace=False)
            for p in range(self.n_particles):
                self._sample_relations(
                    cur_obs, obs_mask,
                    self.E[p],
                    self.R[p],
                    self.var_r[p],
                    sub_relids
                    )
                self._sample_entities(
                    cur_obs,
                    obs_mask,
                    self.E[p],
                    self.R[p],
                    self.var_e[p],
                    sub_entids
                    )

        if self.sample_prior:
            self._sample_prior()

        return self.mean_r[0]

    def predict(self, ids: Sequence[int] = None) -> (numpy.ndarray, None):
        """Predicts labels of instances.

        Notes
        -----
            Unlike in scikit-learn, predictions are always real-valued.
            Predicted labels for a classification problem are represented by
            predicted probabilities of each class.

        Parameters
        ----------
        ids
            List of IDs of instances to predict labels for.

        Returns
        -------
        numpy.ndarray
            An K x D x D  array of corresponding predictions.
        """
        p = multinomial(1, self.p_weights).argmax()

        # reconstruct
        X = numpy.zeros([self.n_relations, self.n_entities, self.n_entities])

        for k in range(self.n_relations):
            X[k] = numpy.dot(numpy.dot(self.E[p], self.R[p][k]), self.E[p].T)

        # logging.critical('R[0, 2,4]: {}'.format(self.R[0,2,4]))

        return X, None

    def reference_predict(self, ids: Sequence[int]) -> (numpy.ndarray, None):
        """Predicts labels using the best possible method.

        Parameters
        ----------
        ids
            List of IDs of instances to predict labels for.

        Returns
        -------
        numpy.ndarray
            An N x 1 x C array of corresponding predictions.
        numpy.ndarray
            A N array of confidences (or None if not applicable).
        """
        return self.predict(ids)

    def _sample_prior(self):
        self._samplevar_r()
        self._samplevar_e()

    def resample(self):
        count = multinomial(self.n_particles, self.p_weights)

        logging.debug("[RESAMPLE] %s", str(count))

        new_E = list()
        new_R = list()

        for p in range(self.n_particles):
            for i in range(count[p]):
                new_E.append(self.E[p].copy())
                new_R.append(self.R[p].copy())

        self.E = numpy.asarray(new_E)
        self.R = numpy.asarray(new_R)
        self.p_weights = numpy.ones(self.n_particles) / self.n_particles

    def compute_particle_weight(self, next_idx, X, mask):
        from scipy.stats import norm
        r_k, e_i, e_j = next_idx

        log_weight = numpy.zeros(self.n_particles)
        for p in range(self.n_particles):

            mean = numpy.dot(
                    numpy.dot(self.E[p][e_i], self.R[p][r_k]),
                    self.E[p][e_j]
                    )
            log_weight[p] = norm.logpdf(X[next_idx], mean, self.var_x)

        log_weight -= numpy.max(log_weight)
        weight = numpy.exp(log_weight)
        weight += 1e-10
        return weight / numpy.sum(weight)

    def _samplevar_r(self):
        for p in range(self.n_particles):
            self.var_r[p] = 1. / gamma(
                0.5 * self.n_relations * self.n_dim * self.n_dim + self.r_alpha,
                1. / (0.5 * numpy.sum(self.R[p] ** 2) + self.r_beta))
        logging.debug("Sampled var_r %.3f", numpy.mean(self.var_r))

    def _samplevar_e(self):
        for p in range(self.n_particles):
            self.var_e[p] = 1. / gamma(
                0.5 * self.n_entities * self.n_dim + self.e_alpha,
                1. / (0.5 * numpy.sum(self.E[p] ** 2) + self.e_beta))
        logging.debug("Sampled var_e %.3f", numpy.mean(self.var_e))

    def _sample_entities(self, X, mask, E, R, var_e, sample_idx=None):
        RE = numpy.zeros([self.n_relations, self.n_entities, self.n_dim])
        RTE = numpy.zeros([self.n_relations, self.n_entities, self.n_dim])

        for k in range(self.n_relations):
            RE[k] = numpy.dot(R[k], E.T).T
            RTE[k] = numpy.dot(R[k].T, E.T).T

        if isinstance(sample_idx, type(None)):
            sample_idx = range(self.n_entities)

        for i in sample_idx:
            self._sample_entity(X, mask, E, R, i, var_e, RE, RTE)
            for k in range(self.n_relations):
                RE[k][i] = numpy.dot(R[k], E[i])
                RTE[k][i] = numpy.dot(R[k].T, E[i])

    def _sample_entity(self, X, mask, E, R, i, var_e, RE=None, RTE=None):
        nz_r = mask[:, i, :].nonzero()
        nz_c = mask[:, :, i].nonzero()
        nnz_r = nz_r[0].size
        nnz_c = nz_c[0].size
        nnz_all = nnz_r + nnz_c

        self.features[:nnz_r] = RE[nz_r]
        self.features[nnz_r:nnz_all] = RTE[nz_c]
        self.xi[:nnz_r] = X[:, i, :][nz_r]
        self.xi[nnz_r:nnz_all] = X[:, :, i][nz_c]
        _xi = self.xi[:nnz_all] * self.features[:nnz_all].T
        xi = numpy.sum(_xi, 1) / self.var_x

        _lambda = numpy.identity(self.n_dim) / var_e
        _lambda += numpy.dot(
            self.features[:nnz_all].T,
            self.features[:nnz_all]) / self.var_x

        # mu = numpy.linalg.solve(_lambda, xi)
        # E[i] = normal(mu, _lambda)

        inv_lambda = numpy.linalg.inv(_lambda)
        mu = numpy.dot(inv_lambda, xi)
        E[i] = multivariate_normal(mu, inv_lambda)

        numpy.mean(numpy.diag(inv_lambda))
        # logging.info('Mean variance E, %d, %f', i, mean_var)

    def _sample_relations(self, X, mask, E, R, var_r, sample_idx=None):
        EXE = numpy.kron(E, E)

        if isinstance(sample_idx, type(None)):
            sample_idx = range(self.n_relations)

        for k in self.valid_relations:
            if k in sample_idx:
                if self.obs_sum[k] != 0:
                    self._sample_relation(X, mask, E, R, k, EXE, var_r)
                else:
                    R[k] = numpy.random.normal(
                        0, var_r, size=[self.n_dim, self.n_dim])

    def _sample_relation(self, X, mask, E, R, k, EXE, var_r):
        _lambda = numpy.identity(self.n_dim ** 2) / var_r
        xi = numpy.zeros(self.n_dim ** 2)

        kron = EXE[mask[k].flatten() == 1]

        if kron.shape[0] != 0:
            _lambda += numpy.dot(kron.T, kron)
            xi += numpy.sum(X[k, mask[k] == 1].flatten() * kron.T, 1)

        _lambda /= self.var_x
        # mu = numpy.linalg.solve(_lambda, xi) / self.var_x

        inv_lambda = numpy.linalg.inv(_lambda)
        mu = numpy.dot(inv_lambda, xi) / self.var_x
        # R[k] = normal(mu, _lambda).reshape([self.n_dim, self.n_dim])
        R[k] = multivariate_normal(
            mu, inv_lambda).reshape([self.n_dim, self.n_dim])
        numpy.mean(numpy.diag(inv_lambda))
        # logging.info('Mean variance R, %d, %f', k, mean_var)

    def _sample_latent_variables(
            self, index, label, mean_e, mean_r, var_e, var_r, E, R):
        '''
        In t^th iteration, we get the new label x_ikj.
        Then update e_i, e_j, r_k using the new label.

        Parameters:
        -------------
        index
            index tuple (k, i, j) of the new label
        label
            label x_kij, 0 or 1.
        mean_e
            mean of entity
        mean_r
            mean of relation
        var_e
            variance of entitiy (sigma_e ^2)
        var_r
            variance of relation (sigma_r ^2)
        E
            posterior of latent entity variable R^{N x D}
            updated from last iteration
        R
            posterior of latent relation variable R^{ K x D x D}
            updated from last iteration
        '''

        # sample relation r_k
        k, i, j = index
        e_i = E[i].reshape(E[i].shape[0], 1)  # D x 1
        e_j = E[j].reshape(E[j].shape[0], 1)  # D x 1
        r_k = R[k]                            # D x D

        mean_ei = mean_e[i].reshape(mean_e[i].shape[0], 1)  # D x 1
        mean_ej = mean_e[j].reshape(mean_e[j].shape[0], 1)  # D x 1
        mean_rk = mean_r[k].reshape(mean_r[k].shape[0], 1)  # D^2 x 1

        '''
        # avoid LinAlgError: Singular matrix error
        m = 10**-6
        var_r[k] += numpy.identity(var_r[k].shape[0]) * m
        var_e[i] += numpy.identity(var_e[i].shape[0]) * m
        var_e[j] += numpy.identity(var_e[j].shape[0]) * m
        '''

        EXE = numpy.kron(e_i, e_j)  # D^2 x 1
        # EXE = EXE.reshape((EXE.shape[0], 1))
        # logging.debug('EXE shape: {}'.format(EXE.shape))
        # logging.debug('var_r[k] shape: {}'.format(var_r[k].shape))

        # D^2 x D^2
        s_rn_inv = numpy.linalg.inv(
            var_r[k]) + numpy.dot(EXE, EXE.T)/self.var_x
        # s_rn_inv += numpy.identity(s_rn_inv.shape[0]) * m
        s_rn = numpy.linalg.inv(s_rn_inv)

        # D^2 x 1
        m_rn = numpy.dot(
            s_rn,
            numpy.dot(numpy.linalg.inv(var_r[k]),
                      mean_rk) + EXE * label/self.var_x
        )

        m_rn = m_rn.reshape(m_rn.shape[0])
        # logging.debug('m_rn shape: {}'.format(m_rn.shape))
        mean_r[k] = m_rn
        var_r[k] = s_rn

        # m_rn = m_rn.reshape(m_rn.shape[0]).tolist()
        R[k] = multivariate_normal(
            m_rn, s_rn).reshape([self.n_dim, self.n_dim])

        # sample entity e_i

        RE = numpy.dot(r_k, e_j)  # D x 1
        # RE = RE.reshape((RE.shape[0], 1))

        # D x D
        s_en_inv = numpy.linalg.inv(var_e[i]) + numpy.dot(RE, RE.T)/self.var_x
        # s_en_inv += numpy.identity(s_en_inv.shape[0]) * m
        s_en = numpy.linalg.inv(s_en_inv)

        # D x 1
        m_en = numpy.dot(
            s_en,
            numpy.dot(numpy.linalg.inv(var_e[i]),
                      mean_ei) + RE * label/self.var_x
        )

        m_en = m_en.reshape(m_en.shape[0])
        mean_e[i] = m_en
        var_e[i] = s_en

        # m_en = m_en.reshape(m_en.shape[0]).tolist()
        E[i] = multivariate_normal(m_en, s_en)

        # sample entity e_j

        ETR = numpy.dot(e_i.T, r_k)  # 1 x D
        # logging.debug('ETR shape: {}'.format(ETR.shape))
        # ETR = ETR.reshape((1, ETR.shape[0])) # 1 x D

        # D x D
        s_en_inv = numpy.linalg.inv(var_e[j]) + numpy.dot(ETR.T, ETR)/self.var_x
        # s_en_inv += numpy.identity(s_en_inv.shape[0]) * m
        # logging.debug('s_en_inv: {}'.format(s_en_inv))
        # logging.debug('ETR: {}'.format(ETR))

        s_en = numpy.linalg.inv(s_en_inv)

        # D x 1
        m_en = numpy.dot(
            s_en,
            numpy.dot(
                numpy.linalg.inv(var_e[j]), mean_ej) + ETR.T * label/self.var_x
                )

        m_en = m_en.reshape(m_en.shape[0])
        mean_e[j] = m_en
        var_e[j] = s_en

        # m_en = m_en.reshape(m_en.shape[0]).tolist()
        E[j] = multivariate_normal(m_en, s_en)

        # return mean_e, mean_r, var_e, var_r
        return s_en_inv, ETR

class UncertaintyTensorPredictor(TensorPredictor):
    """Predict uncertainty for each tensor entry.
       Using uncer(x_ijk) = uncer(e_i)^T uncer(R_k) uncer(e_j)

    Attributes
    ----------
    _db : acton.database.Database
        Database storing features and labels.
    n_particles:
        Number of particles for Thompson sampling.
    n_relations:
        Number of relations (K)
    n_entities:
        Number of entities (N)
    n_dim
        Number of latent dimensions (D)
    var_r
        variance of prior of R
    var_e
        variance of prior of E
    var_x
        variance of X
    sample_prior
        indicates whether sample prior
    E
        P x N x D entity features
    R
        P x K x D x D relation features
    X
        K x N x N labels
    """

    def __init__(self,
                 db: acton.database.Database,
                 n_particles: int = 5,
                 var_r: float = 1.0, var_e: float = 1.0,
                 mean_r: float = 0.0, mean_e: float = 0.0,
                 var_x: float = 0.1,
                 sample_prior: bool = False,
                 n_jobs: int=1
                 ):
        """
        Arguments
        ---------
        db
            Database storing features and labels.
        n_particles:
            Number of particles for Thompson sampling.
        var_r
            variance of prior of R
        var_e
            variance of prior of E
        mean_r
            mean of prior of R
        mean_e
            mean of prior of E
        var_x
            variance of X
        sample_prior
            indicates whether sample prior
        """

        super().__init__(
            db, n_particles, var_r, var_e,
            mean_r, mean_e, var_x, sample_prior, n_jobs)

    def _sample_entity(self, X, mask, E, R, i, var_e, RE=None, RTE=None):
        nz_r = mask[:, i, :].nonzero()
        nz_c = mask[:, :, i].nonzero()
        nnz_r = nz_r[0].size
        nnz_c = nz_c[0].size
        nnz_all = nnz_r + nnz_c

        self.features[:nnz_r] = RE[nz_r]
        self.features[nnz_r:nnz_all] = RTE[nz_c]
        self.xi[:nnz_r] = X[:, i, :][nz_r]
        self.xi[nnz_r:nnz_all] = X[:, :, i][nz_c]
        _xi = self.xi[:nnz_all] * self.features[:nnz_all].T
        xi = numpy.sum(_xi, 1) / self.var_x

        _lambda = numpy.identity(self.n_dim) / var_e
        _lambda += numpy.dot(
            self.features[:nnz_all].T,
            self.features[:nnz_all]) / self.var_x

        # mu = numpy.linalg.solve(_lambda, xi)
        # E[i] = normal(mu, _lambda)

        inv_lambda = numpy.linalg.inv(_lambda)
        mu = numpy.dot(inv_lambda, xi)
        # E[i] = multivariate_normal(mu, inv_lambda)
        # update E using the uncertainty (variance)
        E[i] = inv_lambda

        # numpy.mean(numpy.diag(inv_lambda))
        # logging.info('Mean variance E, %d, %f', i, mean_var)

    def _sample_relation(self, X, mask, E, R, k, EXE, var_r):
        _lambda = numpy.identity(self.n_dim ** 2) / var_r
        xi = numpy.zeros(self.n_dim ** 2)

        kron = EXE[mask[k].flatten() == 1]

        if kron.shape[0] != 0:
            _lambda += numpy.dot(kron.T, kron)
            xi += numpy.sum(X[k, mask[k] == 1].flatten() * kron.T, 1)

        _lambda /= self.var_x
        # mu = numpy.linalg.solve(_lambda, xi) / self.var_x

        inv_lambda = numpy.linalg.inv(_lambda)
        mu = numpy.dot(inv_lambda, xi) / self.var_x
        # R[k] = normal(mu, _lambda).reshape([self.n_dim, self.n_dim])
        # R[k] = multivariate_normal(
        #     mu, inv_lambda).reshape([self.n_dim, self.n_dim])
        # update R using the uncertainty (variance)
        R[k] = inv_lambda.reshape([self.n_dim, self.n_dim])

        numpy.mean(numpy.diag(inv_lambda))
        # logging.info('Mean variance R, %d, %f', k, mean_var)

# Helper functions to generate predictor classes.


def _logistic_regression() -> type:
    return from_class(sklearn.linear_model.LogisticRegression)


def _linear_regression() -> type:
    return from_class(sklearn.linear_model.LinearRegression, regression=True)


def _logistic_regression_committee() -> type:
    def make_committee(db, *args, **kwargs):
        return Committee(_logistic_regression(), db, *args, **kwargs)

    return make_committee


def _kde() -> type:
    return from_class(acton.kde_predictor.KDEClassifier)


PREDICTORS = {
    'LogisticRegression': _logistic_regression(),
    'LogisticRegressionCommittee': _logistic_regression_committee(),
    'LinearRegression': _linear_regression(),
    'KDE': _kde(),
    'GPC': GPClassifier,
    'TensorPredictor': TensorPredictor,
    'UncertaintyTensorPredictor': UncertaintyTensorPredictor
}
