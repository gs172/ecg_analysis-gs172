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

t, v = import_data('test_data28.csv')
t2, v2 = import_data('test_data30.csv')
t3, v3 = import_data('test_data32.csv')


def test_exception1():
    """
    These are 3 unit testing functions for
    the exceptions.

    :return: exceptions if there is one.
    """
    with pytest.raises(ValueError):
        exception(t, v)


def test_exception2():
    with pytest.raises(TypeError):
        exception(t2, v2)


def test_exception3():
    with pytest.raises(OverflowError):
        exception(t3, v3)
