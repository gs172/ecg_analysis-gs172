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

a, b = import_data('test_data2.csv')


@pytest.mark.parametrize("a, b, expected", [
                                        (a, b, 32),
                                        ])
def test_num_beats(a, b, expected):
    """
    This is unit testing for the number of beats
    function.

    :param a: time
    :param b: voltage
    :param expected: expected number of beats

    :return: assertion
    """
    result = num_beats(a, b)
    assert result == 32
