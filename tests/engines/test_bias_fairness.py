"""Tests for the BiasFairness module."""
import numpy as np
import pandas as pd
from pytest import fixture

from ydata_quality.bias_fairness.engine import BiasFairness

@fixture(name='sample_dataset')
def fixture_sample_dataset():
    """Create a sample dataset with both categorical and numerical sensitive features"""
    np.random.seed(42)
    n_samples = 1000
    
    # Create categorical sensitive feature with imbalanced classes
    cat_sensitive = np.random.choice(['A', 'B', 'C'], size=n_samples, p=[0.8, 0.15, 0.05])
    
    # Create numerical sensitive feature with clustered distribution
    num_sensitive = np.concatenate([
        np.random.normal(10, 1, int(n_samples * 0.7)),  # 70% in one cluster
        np.random.normal(20, 1, int(n_samples * 0.2)),  # 20% in another
        np.random.normal(30, 1, n_samples - int(n_samples * 0.9))  # 10% in the last
    ])
    
    # Create other features
    other_feature = np.random.normal(0, 1, n_samples)
    label = np.random.randint(0, 2, n_samples)
    
    return pd.DataFrame({
        'cat_sensitive': cat_sensitive,
        'num_sensitive': num_sensitive,
        'other_feature': other_feature,
        'label': label
    })

@fixture(name='bias_fairness')
def fixture_bias_fairness(sample_dataset):
    """Create BiasFairness instance with sample dataset"""
    sensitive_features = ['cat_sensitive', 'num_sensitive']
    return BiasFairness(df=sample_dataset, 
                       sensitive_features=sensitive_features,
                       label='label',
                       random_state=42)

def test_numerical_representativity_analysis(bias_fairness):
    """Test numerical representativity analysis"""
    result = bias_fairness._analyze_numerical_representativity('num_sensitive', n_clusters=3, threshold=0.2)
    
    assert result is not None
    assert 'cluster_proportions' in result
    assert 'max_difference' in result
    assert 'is_disproportionate' in result
    assert 'cluster_centers' in result
    
    # Check that proportions sum to 1
    assert np.abs(sum(result['cluster_proportions']) - 1.0) < 1e-10
    
    # Check disproportionate distribution is detected
    assert result['is_disproportionate']
    
    # Check cluster centers are returned
    assert len(result['cluster_centers']) == 3

def test_sensitive_representativity(bias_fairness):
    """Test sensitive representativity for both categorical and numerical features"""
    results = bias_fairness.sensitive_representativity(min_pct=0.1, n_clusters=3, num_threshold=0.2)
    
    # Check both features are analyzed
    assert 'cat_sensitive' in results
    assert 'num_sensitive' in results
    
    # Check categorical feature results
    cat_results = results['cat_sensitive']
    assert cat_results['type'] == 'categorical'
    assert isinstance(cat_results['distribution'], pd.Series)
    
    # Check numerical feature results
    num_results = results['num_sensitive']
    assert num_results['type'] == 'numerical'
    assert 'cluster_analysis' in num_results
    
    # Check warnings were generated
    warnings = bias_fairness.get_warnings()
    assert len(warnings) > 0
    
    # Check warning messages
    warning_messages = [w.description for w in warnings]
    assert any('low representativity' in msg for msg in warning_messages)
    assert any('disproportionate representation' in msg for msg in warning_messages)

def test_sensitive_representativity_balanced(sample_dataset):
    """Test sensitive representativity with balanced data"""
    n_samples = 1000
    
    # Create balanced categorical feature
    cat_balanced = np.random.choice(['A', 'B', 'C'], size=n_samples, p=[0.33, 0.33, 0.34])
    
    # Create balanced numerical feature
    num_balanced = np.random.normal(0, 1, n_samples)
    
    balanced_df = pd.DataFrame({
        'cat_balanced': cat_balanced,
        'num_balanced': num_balanced
    })
    
    bf = BiasFairness(df=balanced_df,
                      sensitive_features=['cat_balanced', 'num_balanced'],
                      random_state=42)
    
    results = bf.sensitive_representativity(min_pct=0.1, n_clusters=3, num_threshold=0.3)
    
    # Check no warnings for balanced data
    warnings = bf.get_warnings()
    assert len(warnings) == 0
