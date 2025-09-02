[![](https://api.codacy.com/project/badge/Grade/f1662116e982402c956d2720fbd24507)](https://app.codacy.com/gh/ydataai/ydata-quality)
![](https://img.shields.io/github/workflow/status/ydataai/ydata-quality/release)
![](https://img.shields.io/pypi/status/ydata-quality)
[![](https://pepy.tech/badge/ydata-quality)](https://pypi.org/project/ydata-quality/)
![](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue)
[![](https://img.shields.io/pypi/v/ydata-quality)](https://pypi.org/project/ydata-quality/)
![](https://img.shields.io/github/license/ydataai/ydata-quality)

# YData Quality

ydata_quality is an open-source python library for assessing Data Quality throughout the multiple stages of a data pipeline development. 

A holistic view of the data can only be captured through a look at data from multiple dimensions and `ydata_quality` evaluates it in a modular way wrapped into a single Data Quality engine. This repository contains the core python source scripts and walkthrough tutorials.

## Quickstart

The source code is currently hosted on GitHub at: https://github.com/ydataai/ydata-quality

Binary installers for the latest released version are available at the [Python Package Index (PyPI).](https://pypi.org/project/ydata-quality/)
```
pip install ydata-quality
```

### Comprehensive quality check in few lines of code

```python
from ydata_quality import DataQuality
import pandas as pd

#Load in the data
df = pd.read_csv('./datasets/transformed/census_10k.csv')

# create a DataQuality object from the main class that holds all quality modules
dq = DataQuality(df=df)

# run the tests and outputs a summary of the quality tests
results = dq.evaluate()
```
```
Warnings:
	TOTAL: 5 warning(s)
	Priority 1: 1 warning(s)
	Priority 2: 4 warning(s)

Priority 1 - heavy impact expected:
	* [DUPLICATES - DUPLICATE COLUMNS] Found 1 columns with exactly the same feature values as other columns.
Priority 2 - usage allowed, limited human intelligibility:
	* [DATA RELATIONS - HIGH COLLINEARITY - NUMERICAL] Found 3 numerical variables with high Variance Inflation Factor (VIF>5.0). The variables listed in results are highly collinear with other variables in the dataset. These will make model explainability harder and potentially give way to issues like overfitting. Depending on your end goal you might want to remove the highest VIF variables.
	* [ERRONEOUS DATA - PREDEFINED ERRONEOUS DATA] Found 1960 ED values in the dataset.
	* [DATA RELATIONS - HIGH COLLINEARITY - CATEGORICAL] Found 10 categorical variables with significant collinearity (p-value < 0.05). The variables listed in results are highly collinear with other variables in the dataset and sorted descending according to propensity. These will make model explainability harder and potentially give way to issues like overfitting. Depending on your end goal you might want to remove variables following the provided order.
	* [DUPLICATES - EXACT DUPLICATES] Found 3 instances with exact duplicate feature values.
```


On top of the summary, you can retrieve a list of detected warnings for detailed inspection.
```python
# retrieve a list of data quality warnings 
warnings = dq.get_warnings()
```
## Examples

Here you can find walkthrough tutorials and examples to familiarize with different modules of `ydata_quality`

- [Start Here for Quick and Overall Walkthrough](https://github.com/ydataai/ydata-quality/blob/master/tutorials/main.ipynb)

To dive into any focussed module, and to understand how they work, here are tutorial notebooks:
1. [Bias and Fairness](https://github.com/ydataai/ydata-quality/blob/master/tutorials/bias_fairness.ipynb)
2.  [Data Expectations](https://github.com/ydataai/ydata-quality/blob/master/tutorials/data_expectations.ipynb)
3.  [Data Relations](https://github.com/ydataai/ydata-quality/blob/master/tutorials/data_relations.ipynb)
4.  [Drift Analysis](https://github.com/ydataai/ydata-quality/blob/master/tutorials/drift.ipynb)
5.  [Duplicates](https://github.com/ydataai/ydata-quality/blob/master/tutorials/duplicates.ipynb)
6.  Labelling: [Categoricals](https://github.com/ydataai/ydata-quality/blob/master/tutorials/labelling_categorical.ipynb) and [Numericals](https://github.com/ydataai/ydata-quality/blob/master/tutorials/labelling_numerical.ipynb)
7.  [Missings](https://github.com/ydataai/ydata-quality/blob/master/tutorials/missings.ipynb)
8.  [Erroneous Data](https://github.com/ydataai/ydata-quality/blob/master/tutorials/erroneous_data.ipynb)

## Contributing
We are open to collaboration! If you want to start contributing you only need to:
1.	Search for an issue in which you would like to work on. Issues for newcomers are labeled with good first issue.
2.	Create a PR solving the issue.
3.	We would review every PR and either accept or ask for revisions.

You can also join the discussions on our [Discord Community](https://discord.com/invite/mw7xjJ7b7s) and request features/bug fixes by opening issues on our repository.

## Support
For support in using this library, please join our Discord server. The Discord community is very friendly and great about quickly answering questions about the use and development of the library. [Click here to join our Discord community!](https://discord.com/invite/mw7xjJ7b7s)

## License
[GNU General Public License v3.0](https://github.com/ydataai/ydata-quality/blob/master/LICENSE)

## About

With ♥️ from [YData](https://ydata.ai) [Development team](mailto://developers@ydata.ai)
