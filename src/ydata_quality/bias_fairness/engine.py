"""
Implementation of BiasFairness engine to run bias and fairness analysis.
"""

from typing import List, Optional

from pandas import DataFrame, Series
from dython.nominal import compute_associations

from src.ydata_quality.core.warnings import Priority

from ..core import QualityEngine, QualityWarning
from ..utils.correlations import filter_associations
from ..utils.modelling import (baseline_performance,
                               performance_per_feature_values)


class BiasFairness(QualityEngine):
    """ Engine to run bias and fairness analysis.

    Tests:
        - Proxy Identification: tests for high correlation between sensitive and non-sensitive features
        - Sensitive Predictability: trains a baseline model to predict sensitive attributes
        - Performance Discrimination: checks for performance disparities on sensitive attributes
    """
    # pylint: disable=too-many-arguments
    def __init__(self, df: DataFrame, sensitive_features: List[str], label: Optional[str] = None,
                 random_state: Optional[int] = None, severity: Optional[str] = None):
        """
        Args
            df (DataFrame): reference DataFrame used to run the analysis
            sensitive_features (List[str]): features deemed as sensitive attributes
            label (str, optional): target feature to be predicted
            severity (str, optional): Sets the logger warning threshold to one of the valid levels
                [DEBUG, INFO, WARNING, ERROR, CRITICAL]
        """
        super().__init__(df=df, label=label, random_state=random_state, severity=severity)
        self._sensitive_features = sensitive_features
        self._tests = ["performance_discrimination", "proxy_identification",
                       "sensitive_predictability", "sensitive_representativity"]

    @property
    def sensitive_features(self):
        "Returns a list of sensitive features."
        return self._sensitive_features

    def proxy_identification(self, th=0.5):
        """Tests for non-protected features high correlation with sensitive attributes.

        Non-sensitive features can serve as proxy for protected attributes, exposing the data to a possible
        subsequent bias in the data pipeline. High association values indicate that alternative features can
        be used in place of the original sensitive attributes."""
        # TODO: multiple thresholds per association type (num/num, num/cat, cat/cat)

        # Compute association measures for sensitive features
        corrs = compute_associations(self.df, num_num_assoc='pearson', nom_nom_assoc='cramer')
        corrs = filter_associations(corrs, th=th, name='association', subset=self.sensitive_features)

        if len(corrs) > 0:
            self.store_warning(
                QualityWarning(
                    test=QualityWarning.Test.PROXY_IDENTIFICATION,
                    category=QualityWarning.Category.BIAS_FAIRNESS, priority=Priority.P2, data=corrs,
                    description=f"Found {len(corrs)} feature pairs of correlation "
                    f"to sensitive attributes with values higher than defined threshold ({th})."
                ))
        return corrs

    def sensitive_predictability(self, th=0.5, adjusted_metric=True):
        """Trains a baseline classifier to predict sensitive attributes based on remaining features.

        Good performances indicate that alternative features may be working as proxies for sensitive attributes.
        """
        drop_features = self.sensitive_features + [self.label]  # features to remove in prediction

        performances = Series(index=self.sensitive_features, dtype=str)
        for feat in performances.index:
            data = self.df.drop(columns=[x for x in drop_features if x != feat])  # drop all except target
            performances[feat] = baseline_performance(df=data, label=feat, adjusted_metric=adjusted_metric)

        high_perfs = performances[performances > th]
        if len(high_perfs) > 0:
            self.store_warning(
                QualityWarning(
                    test=QualityWarning.Test.SENSITIVE_ATTRIBUTE_PREDICTABILITY,
                    category=QualityWarning.Category.BIAS_FAIRNESS,
                    priority=Priority.P3, data=high_perfs,
                    description=f"Found {len(high_perfs)} sensitive attribute(s) with high predictability performance"
                    f" (greater than {th})."
                )
            )
        return performances

    def performance_discrimination(self):
        """Checks for performance disparities for sensitive attributes.

        Get the performance of a baseline model for each feature value of a sensitive attribute.
        High disparities in the performance metrics indicate that the model may not be fair across sensitive attributes.
        """
        # TODO: support error rate parity metrics (e.g. false positive rate, positive rate)
        if self.label is None:
            self._logger.warning(
                'Argument "label" must be defined to calculate performance discrimination metric. Skipping test.')

        res = {}
        for feat in self.sensitive_features:
            res[feat] = Series(performance_per_feature_values(df=self.df, feature=feat, label=self.label))
        return res

    def sensitive_representativity(self, min_pct: float = 0.01):
        """Checks categorical sensitive attributes minimum representativity of feature values.

        Raises a warning if a feature value of a categorical sensitive attribute is not represented above a min_pct \
percentage.
        """
        # TODO: Representativity for numerical features
        res = {}
        categorical_sensitives = [
            k for (
                k,
                v) in self.dtypes.items() if (
                v == 'categorical') & (
                k in self.sensitive_features)]
        for cat in categorical_sensitives:
            dist = self.df[cat].value_counts(normalize=True)  # normalized presence of feature values
            res[cat] = dist  # store the distribution
            low_dist = dist[dist < min_pct]  # filter for low representativity
            if len(low_dist) > 0:
                self.store_warning(
                    QualityWarning(
                        test=QualityWarning.Test.SENSITIVE_ATTRIBUTE_REPRESENTATIVITY,
                        category=QualityWarning.Category.BIAS_FAIRNESS, priority=Priority.P2, data=low_dist,
                        description=f"Found {len(low_dist)} values of '{cat}' \
sensitive attribute with low representativity in the dataset (below {min_pct*100:.2f}%)."
                    )
                )
        return res