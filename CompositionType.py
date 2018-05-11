from enum import Enum


class CompositionType(Enum):
    """Defines the type of majority voting strategy


    """
    MEAN = 'Mean'
    MEDIAN = 'Median'
