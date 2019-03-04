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
from ECG_analysis import save_patient
from ECG_analysis import runcode

t, v = import_data('test_data1.csv')
dur = duration(t)
max_v, min_v = voltage_extremes(v)
number_of_beats = num_beats(t, v)
mean_hr_bpm = bpm(dur, number_of_beats)
peaks_array = beats(t, v)
patient = create_patient(dur, max_v, min_v, number_of_beats,
                         mean_hr_bpm, peaks_array)
out_file = save_patient(patient)
type_of_file = type(out_file)


@pytest.mark.parametrize("a, expected", [
                                        (patient, type_of_file),
                                        ])
def test_create_patient(a, expected):
    """
    Unit testing for create_patient function to see
    if a JSON file is the output.

    :param a: patient dictionary
    :param expected: JSON type

    :return: assertion
    """
    result = type(save_patient(a))
    assert result == expected
