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


@pytest.mark.parametrize("a, expected_max, expected_min", [
                                        (v, 0.60625, -0.025),
                                        ])
def test_voltage_extremes(a, expected_max, expected_min):
    """
    This is unit testing for the max and min voltages.

    :param a: voltage
    :param expected_max: max voltage
    :param expected_min: min voltage

    :return: assertions
    """
    result, result1 = voltage_extremes(a)
    result = float(result)
    result1 = float(result1)
    assert result == expected_max
    assert result1 == expected_min
