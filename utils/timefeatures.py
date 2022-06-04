from typing import List

import numpy as np
import pandas as pd

class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5

class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5

class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5

class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


def time_features_from_frequency_str() -> List[TimeFeature]:
    feature_classes = [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear]
    return [cls() for cls in feature_classes]

def time_features(dates):
    # 年、月、日、小时、分钟、秒都被缩放到[-0.5,0.5]区间
    dates = pd.to_datetime(dates.date.values)
    return np.vstack([feat(dates) for feat in time_features_from_frequency_str()]).transpose(1,0)
