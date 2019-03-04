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


@pytest.mark.parametrize("a, b, expected", [
                                        (t, v, [0.043, 0.793, 1.543, 2.293,
                                                3.043, 3.793, 4.543, 5.293,
                                                6.043, 6.793, 7.543, 8.293,
                                                9.043, 9.793, 10.543, 11.293,
                                                12.043, 12.793, 13.543]),
                                        ])
def test_beats(a, b, expected):
    """
    This function tests the beats function by seeing
    if the resulting time array is correct for a file.

    :param a: time
    :param b: voltage
    :param expected: expected output

    :return: assertion
    """
    result = beats(a, b)
    result = [float(val) for val in result]
    assert result == [0.043, 0.793, 1.543, 2.293, 3.043,
                      3.793, 4.543, 5.293, 6.043, 6.793, 7.543, 8.293,
                      9.043, 9.793, 10.543, 11.293, 12.043, 12.793, 13.543]
