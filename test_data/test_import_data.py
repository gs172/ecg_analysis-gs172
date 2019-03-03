import pytest
import csv
import numpy as np
from scipy.signal import find_peaks
from scipy.signal import butter, lfilter
from ECG_analysis import import_data
from ECG_analysis import exception
from ECG_analysis import nan_helper
from ECG_analysis import butter_bandpass
from ECG_analysis import butter_bandpass_filter
from ECG_analysis import duration
from ECG_analysis import voltage_extremes
from ECG_analysis import num_beats
from ECG_analysis import beats
from ECG_analysis import bpm
from ECG_analysis import create_patient
from ECG_analysis import runcode

t, v = import_data('test_data21.csv')


@pytest.mark.parametrize("a, expected1, expected2", [
                                        ('test_data21.csv', t, v),
                                        ])
def test_import_data(a, expected1, expected2):
    """
    This is unit testing for import data function.

    :param a: file name
    :param expected1: time
    :param expected2: voltage

    :return: the two vectors
    """
    result, result1 = import_data(a)
    assert result == t
    assert result1 == v
