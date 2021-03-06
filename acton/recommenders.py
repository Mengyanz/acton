"""Recommender classes."""

from abc import ABC, abstractmethod
import logging
from typing import Sequence
import warnings

import acton.database
import numpy
import scipy.stats

_E_ALPHA = 1.
_E_BETA = 1.
_R_ALPHA = 1.
_R_BETA = 1.
_P_SAMPLE_GAP = 5
_P_SAMPLE = False
_PARALLEL = False
_MAX_THREAD = 4
_POS_VAL = 1
_MC_MOVE = 1
_SGLD = False
_NMINI = 1
_GIBBS_INIT = True
_SAMPLE_ALL = True

_VAR_E = 1.
_VAR_R = 1.
_VAR_X = 0.01

_DEST = ''
_LOG = ''

a = 0.001
b = 0.01
tau = -0.55

MIN_VAL = numpy.iinfo(numpy.int32).min


def choose_mmr(features: numpy.ndarray, scores: numpy.ndarray, n: int,
               l: float=0.5) -> Sequence[int]:
    """Chooses n scores using maximal marginal relevance.

    Notes
    -----
    Scores are chosen from highest to lowest. If there are less scores to choose
    from than requested, all scores will be returned in order of preference.

    Parameters
    ----------
    scores
        1D array of scores.
    n
        Number of scores to choose.
    l
        Lambda parameter for MMR. l = 1 gives a relevance-ranked list and l = 0
        gives a maximal diversity ranking.

    Returns
    -------
    Sequence[int]
        List of indices of scores chosen.
    """
    if n < 0:
        raise ValueError('n must be a non-negative integer.')

    if n == 0:
        return []

    selections = [scores.argmax()]
    selections_set = set(selections)

    logging.debug('Running MMR.')
    dists = []
    dists_matrix = None
    while len(selections) < n:
        if len(selections) % (n // 10) == 0:
            logging.debug('MMR epoch {}/{}.'.format(len(selections), n))
        # Compute distances for last selection.
        last = features[selections[-1]:selections[-1] + 1]
        last_dists = numpy.linalg.norm(features - last, axis=1)
        dists.append(last_dists)
        dists_matrix = numpy.array(dists)

        next_best = None
        next_best_margin = float('-inf')

        for i in range(len(scores)):
            if i in selections_set:
                continue

            margin = l * (scores[i] - (1 - l) * dists_matrix[:, i].max())
            if margin > next_best_margin:
                next_best_margin = margin
                next_best = i

        if next_best is None:
            break

        selections.append(next_best)
        selections_set.add(next_best)

    return selections


def choose_boltzmann(features: numpy.ndarray, scores: numpy.ndarray, n: int,
                     temperature: float=1.0) -> Sequence[int]:
    """Chooses n scores using a Boltzmann distribution.

    Notes
    -----
    Scores are chosen from highest to lowest. If there are less scores to choose
    from than requested, all scores will be returned in order of preference.

    Parameters
    ----------
    scores
        1D array of scores.
    n
        Number of scores to choose.
    temperature
        Temperature parameter for sampling. Higher temperatures give more
        diversity.

    Returns
    -------
    Sequence[int]
        List of indices of scores chosen.
    """
    if n < 0:
        raise ValueError('n must be a non-negative integer.')

    if n == 0:
        return []

    boltzmann_scores = numpy.exp(scores / temperature)
    boltzmann_scores /= boltzmann_scores.sum()
    not_chosen = list(range(len(boltzmann_scores)))
    chosen = []
    while len(chosen) < n and not_chosen:
        scores_ = boltzmann_scores[not_chosen]
        r = numpy.random.uniform(high=scores_.sum())
        total = 0
        upto = 0
        while True:
            score = scores_[upto]
            total += score
            if total > r:
                break

            upto += 1
        chosen.append(not_chosen[upto])
        not_chosen.pop(upto)

    return chosen


class Recommender(ABC):
    """Base class for recommenders.

    Attributes
    ----------
    """

    @abstractmethod
    def recommend(self, ids: Sequence[int],
                  predictions: numpy.ndarray,
                  n: int=1, diversity: float=0.5) -> Sequence[int]:
        """Recommends an instance to label.

        Parameters
        ----------
        ids
            Sequence of IDs in the unlabelled data pool.
        predictions
            N x T x C array of predictions.
        n
            Number of recommendations to make.
        diversity
            Recommendation diversity in [0, 1].

        Returns
        -------
        Sequence[int]
            IDs of the instances to label.
        """


class RandomRecommender(Recommender):
    """Recommends instances at random."""

    def __init__(self, db: acton.database.Database):
        """
        Parameters
        ----------
        db
            Features database.
        """
        self._db = db

    def recommend(self, ids: Sequence[int],
                  predictions: numpy.ndarray,
                  n: int=1, diversity: float=0.5) -> Sequence[int]:
        """Recommends an instance to label.

        Parameters
        ----------
        ids
            Sequence of IDs in the unlabelled data pool.
        predictions
            N x T x C array of predictions.
        n
            Number of recommendations to make.
        diversity
            Recommendation diversity in [0, 1].

        Returns
        -------
        Sequence[int]
            IDs of the instances to label.
        """
        return numpy.random.choice(list(ids), size=n)


class QBCRecommender(Recommender):
    """Recommends instances by committee disagreement."""

    def __init__(self, db: acton.database.Database):
        """
        Parameters
        ----------
        db
            Features database.
        """
        self._db = db

    def recommend(self, ids: Sequence[int],
                  predictions: numpy.ndarray,
                  n: int=1, diversity: float=0.5) -> Sequence[int]:
        """Recommends an instance to label.

        Notes
        -----
        Assumes predictions are probabilities of positive binary label.

        Parameters
        ----------
        ids
            Sequence of IDs in the unlabelled data pool.
        predictions
            N x T x C array of predictions. The ith row must correspond with the
            ith ID in the sequence.
        n
            Number of recommendations to make.
        diversity
            Recommendation diversity in [0, 1].

        Returns
        -------
        Sequence[int]
            IDs of the instances to label.
        """
        assert predictions.shape[1] > 2, "QBC must have > 2 predictors."
        assert len(ids) == predictions.shape[0]
        assert 0 <= diversity <= 1
        labels = predictions.argmax(axis=2)
        plurality_labels, plurality_counts = scipy.stats.mode(labels, axis=1)
        assert plurality_labels.shape == (predictions.shape[0], 1), \
            'plurality_labels has shape {}; expected {}'.format(
                plurality_labels.shape, (predictions.shape[0], 1))
        agree_with_plurality = labels == plurality_labels
        assert labels.shape == agree_with_plurality.shape
        n_agree = labels.sum(axis=1)
        p_agree = n_agree / n_agree.max()  # Agreement is now between 0 and 1.
        disagreement = 1 - p_agree
        indices = choose_boltzmann(self._db.read_features(ids), disagreement, n,
                                   temperature=diversity * 2)
        return [ids[i] for i in indices]


class UncertaintyRecommender(Recommender):
    """Recommends instances by confidence-based uncertainty sampling."""

    def __init__(self, db: acton.database.Database):
        """
        Parameters
        ----------
        db
            Features database.
        """
        self._db = db

    def recommend(self, ids: Sequence[int],
                  predictions: numpy.ndarray,
                  n: int=1, diversity: float=0.5) -> Sequence[int]:
        """Recommends an instance to label.

        Notes
        -----
        Assumes predictions are probabilities of positive binary label.

        Parameters
        ----------
        ids
            Sequence of IDs in the unlabelled data pool.
        predictions
            N x 1 x C array of predictions. The ith row must correspond with the
            ith ID in the sequence.
        n
            Number of recommendations to make.
        diversity
            Recommendation diversity in [0, 1].

        Returns
        -------
        Sequence[int]
            IDs of the instances to label.
        """
        if predictions.shape[1] != 1:
            raise ValueError('Uncertainty sampling must have one predictor')

        assert len(ids) == predictions.shape[0]

        # x* = argmax (1 - p(y^ | x)) where y^ = argmax p(y | x) (Settles 2009).
        proximities = 1 - predictions.max(axis=2).ravel()
        assert proximities.shape == (len(ids),)

        indices = choose_boltzmann(self._db.read_features(ids), proximities, n,
                                   temperature=diversity * 2)
        return [ids[i] for i in indices]


class EntropyRecommender(Recommender):
    """Recommends instances by confidence-based uncertainty sampling."""

    def __init__(self, db: acton.database.Database):
        """
        Parameters
        ----------
        db
            Features database.
        """
        self._db = db

    def recommend(self, ids: Sequence[int],
                  predictions: numpy.ndarray,
                  n: int=1, diversity: float=0.5) -> Sequence[int]:
        """Recommends an instance to label.

        Parameters
        ----------
        ids
            Sequence of IDs in the unlabelled data pool.
        predictions
            N x 1 x C array of predictions. The ith row must correspond with the
            ith ID in the sequence.
        n
            Number of recommendations to make.
        diversity
            Recommendation diversity in [0, 1].

        Returns
        -------
        Sequence[int]
            IDs of the instances to label.
        """
        if predictions.shape[1] != 1:
            raise ValueError('Uncertainty sampling must have one predictor')

        assert len(ids) == predictions.shape[0]

        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', category=RuntimeWarning)
            proximities = -predictions * numpy.log(predictions)

        proximities = proximities.sum(axis=1).max(axis=1).ravel()
        proximities[numpy.isnan(proximities)] = float('-inf')

        assert proximities.shape == (len(ids),)

        indices = choose_boltzmann(self._db.read_features(ids), proximities, n,
                                   temperature=diversity * 2)
        return [ids[i] for i in indices]


class MarginRecommender(Recommender):
    """Recommends instances by margin-based uncertainty sampling."""

    def __init__(self, db: acton.database.Database):
        """
        Parameters
        ----------
        db
            Features database.
        """
        self._db = db

    def recommend(self, ids: Sequence[int],
                  predictions: numpy.ndarray,
                  n: int=1, diversity: float=0.5) -> Sequence[int]:
        """Recommends an instance to label.

        Notes
        -----
        Assumes predictions are probabilities of positive binary label.

        Parameters
        ----------
        ids
            Sequence of IDs in the unlabelled data pool.
        predictions
            N x 1 x C array of predictions. The ith row must correspond with the
            ith ID in the sequence.
        n
            Number of recommendations to make.
        diversity
            Recommendation diversity in [0, 1].

        Returns
        -------
        Sequence[int]
            IDs of the instances to label.
        """
        if predictions.shape[1] != 1:
            raise ValueError('Uncertainty sampling must have one predictor')

        assert len(ids) == predictions.shape[0]

        # x* = argmin p(y1^ | x) - p(y2^ | x) where yn^ = argmax p(yn | x)
        # (Settles 2009).
        partitioned = numpy.partition(predictions, -2, axis=2)
        most_likely = partitioned[:, 0, -1]
        second_most_likely = partitioned[:, 0, -2]
        assert most_likely.shape == (len(ids),)
        scores = 1 - (most_likely - second_most_likely)

        indices = choose_boltzmann(self._db.read_features(ids), scores, n,
                                   temperature=diversity * 2)
        return [ids[i] for i in indices]


class ThompsonSamplingRecommender(Recommender):
    """Recommends instances by Thompson Sampling.
       Input:
           K x N x N predictions.
       Output
           IDs of the instances to label.

       Only support one recommendation

    Attributes
    -----------------
    db
        Features database.

    """

    def __init__(self, db: acton.database.Database):
        """
        Parameters
        ----------
        db
            Features database.
        """
        self._db = db

    def recommend(self, ids: Sequence[tuple],
                  predictions: numpy.ndarray,
                  n: int=1, diversity: float=0.0,
                  repreated_labelling: bool = True) -> Sequence[int]:
        """Recommends an instance to label.

        Notes
        -----
        Predictions are reconstruct enties equal to e_i R_k e_j^T.

        Parameters
        ----------
        ids
            Sequence of IDs in the unlabelled data pool.
        predictions
            K x N x N array of predictions.
        n
            Number of recommendations to make.
        diversity
            recommend methods selection.
            0.5 represents Thompson Samplig;
            1.0 represents Random Sampling
        repeated_labelling
            whether allow one instance to be labelled more than once

        Returns
        -------
        Sequence[int]
            IDs of the instances to label.
        """

        n_relations, n_entities, _ = predictions.shape

        MIN_VAL = numpy.iinfo(numpy.int32).min

        # mask tensor: 0 represents unlabelled, 1 represents labelled

        if repreated_labelling:
            # test: allow repeated labelling
            mask = numpy.zeros_like(predictions)
        else:
            mask = numpy.ones_like(predictions)
            for _tuple in ids:
                r_k, e_i, e_j = _tuple
                mask[r_k, e_i, e_j] = 0

        if diversity == 0.0:
            predictions[mask == 1] = MIN_VAL
            return [numpy.unravel_index(predictions.argmax(),
                    predictions.shape)]
        else:
            correct = False
            while not correct:
                sample = (numpy.random.randint(n_relations),
                          numpy.random.randint(n_entities),
                          numpy.random.randint(n_entities))
                if mask[sample] == 0:
                    correct = True
            return [sample]


# For safe string-based access to recommender classes.
RECOMMENDERS = {
    'RandomRecommender': RandomRecommender,
    'QBCRecommender': QBCRecommender,
    'UncertaintyRecommender': UncertaintyRecommender,
    'EntropyRecommender': EntropyRecommender,
    'MarginRecommender': MarginRecommender,
    'ThompsonSamplingRecommender': ThompsonSamplingRecommender,
    'None': RandomRecommender,
}
