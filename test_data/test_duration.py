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


@pytest.mark.parametrize("a,expected", [
                                        (t, 13.887),
                                        ])
def test_duration(a, expected):
    """
    This is unit testing for test_duration.

    :param a: time
    :param expected: expected duration

    :return: assertion
    """
    result = duration(a)
    assert result == expected
