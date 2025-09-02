"""
Custom implementations of Enums.
"""

from enum import Enum


class PredictionTask(Enum):
    "Enum of supported prediction tasks."
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'


class DataFrameType(Enum):
    "Enum of supported dataset types."
    TABULAR = 'tabular'
    TIMESERIES = 'timeseries'


class OrderedEnum(Enum):
    "Enum with support for ordering."

    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented

    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


class StringEnum(Enum):
    """Enum that allows case-insensitive string lookup."""

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            upper_value = value.upper()

            key = cls.find_member(upper_value)
            if key is not None:
                return key

            lower_value = value.lower()

            key = cls.find_member(lower_value)
            if key is not None:
                return key

        raise ValueError(f"{value} is not a valid {cls.__name__}")

    @classmethod
    def find_member(cls, value: str):
        """Find an enum member by its string value.

        Args:
            value: The string value to look up

        Returns:
            The enum member if found, None otherwise
        """
        if value in cls.__members__:
            return cls(value)

        return None
