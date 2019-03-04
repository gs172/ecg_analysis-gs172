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

t, v = import_data('test_data1.csv')
dur = duration(t)
max_v, min_v = voltage_extremes(v)
number_of_beats = num_beats(t, v)
mean_hr_bpm = bpm(dur, number_of_beats)
peaks_array = beats(t, v)
patient = create_patient(dur, max_v, min_v, number_of_beats,
                         mean_hr_bpm, peaks_array)


@pytest.mark.parametrize("a, b, c, d, e, f, expected", [
                                        (dur, max_v, min_v, number_of_beats,
                                         mean_hr_bpm, peaks_array, patient),
                                        ])
def test_create_patient(a, b, c, d, e, f, expected):
    """
    Unit test for create_patient.py to see if function
    generates appropriate dictionary.

    :param a: dur
    :param b: max_v
    :param c: min_v
    :param d: number of beats
    :param e: bpm
    :param f: peaks array
    :param expected: expected

    :return: assertion
    """
    result = create_patient(a, b, c, d, e, f)
    assert result == expected
