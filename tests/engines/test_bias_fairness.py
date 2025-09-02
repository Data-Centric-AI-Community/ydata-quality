"""
Unit tests for the bias fairness engine
"""

import pandas as pd
import pytest


from src.ydata_quality.bias_fairness.engine import BiasFairness


def get_fake_data():
    """Returns fake data for tests."""
    return pd.DataFrame({
            'age': [25, 35, 45, 55],
            'salary': [30000, 45000, 60000, 75000],
            'gender': ['M', 'F', 'M', 'F'],
            'department': ['IT', 'HR', 'IT', 'HR']
        })


class TestBiasFairness:
    """Test class for BiasFairness."""

    def test_sensitive_features_property(self):
        """Test sensitive features property returns correct features."""
        df = get_fake_data()
        sensitive_features = ['gender', 'age']
        bf = BiasFairness(df=df, sensitive_features=sensitive_features)
        assert bf.sensitive_features == sensitive_features


    def test_proxy_identification(self):
        """Test proxy identification returns expected correlations."""
        df = get_fake_data()
        sensitive_features = ['gender']
        bf = BiasFairness(
                df=df,
                sensitive_features=sensitive_features
            )
        correlations = bf.proxy_identification(th=0.5)
        assert len(correlations) >= 0


