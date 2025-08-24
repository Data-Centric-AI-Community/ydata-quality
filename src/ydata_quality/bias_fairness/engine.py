"""
Implementation of BiasFairness engine to run bias and fairness analysis.
"""

from typing import Dict, List, Optional, Union
import numpy as np
from pandas import DataFrame, Series
from dython.nominal import associations
from sklearn.cluster import KMeans

from ..core.warnings import Priority

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
        
    def _analyze_numerical_representativity(self, feature: str, n_clusters: int = 5, threshold: float = 0.2) -> Dict:
        """Analyzes representativity of numerical sensitive features using clustering.
        
        Args:
            feature: Name of the numerical feature to analyze
            n_clusters: Number of clusters for grouping
            threshold: Maximum allowed representativity difference between clusters
            
        Returns:
            Dict with cluster analysis results
        """
        data = self.df[[feature]].copy()
        
        # Handle NaN values
        data = data.dropna()
        if len(data) == 0:
            return None
            
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=self._random_state)
        clusters = kmeans.fit_predict(data)
        
        # Calculate cluster proportions
        proportions = np.bincount(clusters) / len(clusters)
        
        # Check for disproportionate representation
        max_diff = np.max(proportions) - np.min(proportions)
        
        return {
            'cluster_proportions': proportions.tolist(),
            'max_difference': max_diff,
            'is_disproportionate': max_diff > threshold,
            'cluster_centers': kmeans.cluster_centers_.flatten().tolist()
        }

    def proxy_identification(self, th=0.5):
        """Tests for non-protected features high correlation with sensitive attributes.

        Non-sensitive features can serve as proxy for protected attributes, exposing the data to a possible
        subsequent bias in the data pipeline. High association values indicate that alternative features can
        be used in place of the original sensitive attributes.
        """
        # TODO: multiple thresholds per association type (num/num, num/cat, cat/cat)

        # Compute association measures for sensitive features
        corrs = associations(self.df, num_num_assoc='pearson', nom_nom_assoc='cramer', compute_only=True)['corr']
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

    def sensitive_representativity(self, min_pct: float = 0.01, n_clusters: int = 5, num_threshold: float = 0.2):
        """Checks sensitive attributes representativity for both categorical and numerical features.

        For categorical features:
            Raises a warning if a feature value is not represented above a min_pct percentage.
            
        For numerical features:
            Uses clustering to group values and checks if clusters have balanced representation.
            Raises a warning if the difference in cluster sizes exceeds the threshold.

        Args:
            min_pct: Minimum percentage for categorical feature values
            n_clusters: Number of clusters for numerical features
            num_threshold: Maximum allowed difference in cluster sizes for numerical features
        """
        res = {}
        
        for feature in self.sensitive_features:
            if self.dtypes[feature] == 'categorical':
                # Handle categorical features
                dist = self.df[feature].value_counts(normalize=True)
                res[feature] = {'type': 'categorical', 'distribution': dist}
                
                low_dist = dist[dist < min_pct]
                if len(low_dist) > 0:
                    self.store_warning(
                        QualityWarning(
                            test=QualityWarning.Test.SENSITIVE_ATTRIBUTE_REPRESENTATIVITY,
                            category=QualityWarning.Category.BIAS_FAIRNESS,
                            priority=Priority.P2,
                            data=low_dist,
                            description=f"Found {len(low_dist)} values of '{feature}' \
sensitive attribute with low representativity in the dataset (below {min_pct*100:.2f}%)."
                        )
                    )
            else:
                # Handle numerical features
                cluster_analysis = self._analyze_numerical_representativity(
                    feature, n_clusters=n_clusters, threshold=num_threshold
                )
                
                if cluster_analysis:
                    res[feature] = {'type': 'numerical', 'cluster_analysis': cluster_analysis}
                    
                    if cluster_analysis['is_disproportionate']:
                        self.store_warning(
                            QualityWarning(
                                test=QualityWarning.Test.SENSITIVE_ATTRIBUTE_REPRESENTATIVITY,
                                category=QualityWarning.Category.BIAS_FAIRNESS,
                                priority=Priority.P2,
                                data=cluster_analysis,
                                description=f"Numerical sensitive attribute '{feature}' shows disproportionate \
representation across value ranges. Maximum difference between cluster sizes: {cluster_analysis['max_difference']:.2f}"
                            )
                        )
        
        return res
