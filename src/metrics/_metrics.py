import numpy as np
import pandas as pd


__all__ = [
    "smape"
]


def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))
