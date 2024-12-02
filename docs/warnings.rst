========
Warnings
========

Structure
---------
A QualityWarning contains all the necessary data required for a Data Scientist to understand and assess the impact of a given data quality issue found during the data quality evaluation.

A QualityWarning is composed by:
    * Category: name of the main test suite (e.g. Duplicates, Bias&Fairness)
    * Test: name of the individual test (e.g. Exact Duplicates, Performance Bias)
    * Description: long-text description of the data quality details
    * Priority: expected impact of the data quality warning
    * Data: sample of data that showcases the data quality warning

Priorities
----------
The Priority aims to provide a quick and intuitive level of severity of a QualityWarning.

========    ============
Priority    Description
========    ============
P0          Blocker. Critical issues that block using the dataset.
P1          High. Heavy impact expected on downstream application.
P2          Medium. Allows usage but may block human-intelligible insights.
P3          Low. Minor impact, aesthetic. No impact on downstream application.
========    ============

Technically, a Priority is implemented as an OrderedEnum so that we can apply comparison operators (<, <=, >, >=). More details on OrderedEnum are available in the utils sub-package.

