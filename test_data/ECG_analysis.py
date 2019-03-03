import csv
import numpy as np
from numpy import isnan
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import butter, lfilter
from scipy.signal import freqz
import logging

logging.basicConfig(filename="ECG_analysis.log", filemode="w",
                    level=logging.INFO)


def import_data(file):
    """
    This function reads in a csv file of ECG data and outputs the time
    and voltage arrays as lists.

    :param file: inputs a csv file name

    :return: time and voltage array of the ECG signal
    """
    t = []
    v = []
    with open(file, 'r') as csvfile:
        for row in csv.reader(csvfile, delimiter=','):
            t.append(row[0])
            v.append(row[1])
        return t, v


def exception(t, v):
    """
    This function tests for various types of exceptions that can occur
    in a given input ECG signal

    :param t: time array
    :param v: voltage array

    :return: the time and voltage arrays and possible exceptions
    """
    if "NaN" in str(t) or "NaN" in str(v):
        raise ValueError("NaN exists in list")
    if "bad data" in str(t) or "bad data" in str(v):
        raise TypeError("Bad data type in list")
    if "300" in str(v):
        raise OverflowError("Voltage exceeds 300mV!")
    if " " in str(t) or " " in str(v):
        raise SyntaxError("Empty space in data set")
    return t, v


def nan_helper(y):
    """
    This function is taken from
    https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
    to facilitate interpolation between array values that have NaN.

    :param y: some array with NaN values

    :return: logical indices of NaNs in the array and converted indices
    """
    return np.isnan(y), lambda z: z.nonzero()[0]


def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    This function is taken from
    https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
    to create a bandpass Butterworth Filter.

    :param lowcut: lower cutoff frequency
    :param highcut: higher cutoff frequency
    :param fs: signal frequency
    :param order: order of the filter

    :return: filter parameters a and b
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    This function is taken from the same source as the butterworth filter.
    This function is used tp then apply the generated filter to the data.

    :param data: data array to be filtered
    :param lowcut: lower cutoff frequency
    :param highcut: higher cutoff frequency
    :param fs: signal frequency
    :param order: order of the filter

    :return: the filtered signal
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def duration(t):
    """
    This function calculates the duration of an ECG signal by
    subtracting the first time point from the final time point.

    :param t: time vector

    :return: the duration of the signal
    """
    t = [float(val) for val in t]
    last_t = t[-1]
    first_t = t[0]
    dur = last_t - first_t
    return dur


def voltage_extremes(v):
    """
    This function finds the max and min voltage values of the ECG.

    :param v: voltage array

    :return: max and min voltage values
    """
    max_v = (max(v))
    min_v = (min(v))
    return max_v, min_v


def num_beats(t, v):
    """
    This function applies the filter to the ECG signal and performs
    a peak detection scheme to determine the number of beats in the signal.

    :param t: time array
    :param v: voltage array

    :return: number of detected beats
    """
    fs = 360
    lowcut = 10
    highcut = 60
    dur = duration(t)
    v = [float(val) for val in v]
    for order in [3, 6, 9]:
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = butter_bandpass_filter(v, lowcut, highcut, fs, order=6)
    peaks, _ = find_peaks(v, distance=250)
    if 5 < len(peaks) < 20 and dur < 35:
        peaks1, _ = find_peaks(v, distance=250)
        number_of_beats = len(peaks1)
    elif len(peaks) > 20 and dur < 35:
        peaks1, _ = find_peaks(y, distance=250)
        number_of_beats = len(peaks1)
    elif len(peaks) > 20 and dur > 35:
        peaks1, _ = find_peaks(y, distance=120)
        number_of_beats = len(peaks1)
    return number_of_beats


def beats(t, v):
    """
    This function takes the time and voltage arguments and filters
    them. It then returns the array of time points at which a beat
    is detected.

    :param t: time array
    :param v: voltage array

    :return: an array of time points at which beats occurs
    """
    fs = 360
    lowcut = 10
    highcut = 60
    dur = duration(t)
    v = [float(val) for val in v]
    for order in [3, 6, 9]:
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = butter_bandpass_filter(v, lowcut, highcut, fs, order=6)
    peaks, _ = find_peaks(v, distance=250)
    if 5 < len(peaks) < 20 and dur < 35:
        peaks, _ = find_peaks(v, distance=250)
        peaks = np.round(peaks)
        peaks = peaks.tolist()
        peaks = [int(i) for i in peaks]
        peaks_array = [t[i] for i in peaks]
    elif len(peaks) > 20 and dur < 35:
        peaks, _ = find_peaks(y, distance=250)
        peaks = np.round(peaks)
        peaks = peaks.tolist()
        peaks = [int(i) for i in peaks]
        peaks_array = [t[i] for i in peaks]
    elif len(peaks) > 20 and dur > 35:
        peaks, _ = find_peaks(y, distance=120)
        peaks = np.round(peaks)
        peaks = peaks.tolist()
        peaks = [int(i) for i in peaks]
        peaks_array = [t[i] for i in peaks]
    return peaks_array


def bpm(dur, number_of_beats):
    """
    This function calculates the beats per minute. It does
    so by taking the number of beats over the duration of the signal
    and multiplying it by 60 to get the beats per minute value.

    :param dur: duration of the signal
    :param number_of_beats:  number of detected beats
    :return: mean beats per minute
    """
    mean_hr_bpm = number_of_beats*60/dur
    mean_hr_bpm = round(mean_hr_bpm)
    return mean_hr_bpm


def create_patient(dur, max_v, min_v, number_of_beats, mean_hr_bpm,
                   peaks_array):
    """
    This function creates a dictionary for each patient generated.

    :param dur: duration of signal
    :param max_v: max voltage
    :param min_v: min voltage
    :param number_of_beats: number of detected beats
    :param mean_hr_bpm: mean calculated beats per minute
    :param peaks_array: array of detected beats time points

    :return: patient dictionary
    """
    patient = {
        'Duration': dur,
        'Maximum Lead Voltage': max_v,
        'Minimum Lead Voltage': min_v,
        'Number of Detected Beats': number_of_beats,
        'Average Heart Rate': mean_hr_bpm,
        'Array of Beat Occurances': peaks_array
    }
    return patient


def save_patient(patient):
    """
    This function takes the patient dictionary and outputs it
    into a JSON format file for each patient.

    :param patient: patient dictionary generated

    :return: output JSON file
    """
    out_file = open('test_data1.json', 'w')
    import json
    json.dump(patient, out_file)
    out_file.close()
    return


def runcode():
    """
    This is the caller function that runs all the modules above.
    It also accounts for all the potential exceptions and tries various
    ways to overcome them.

    :return: runs the functions
    """
    t, v = import_data('test_data1.csv')
    logging.info("Starting new analysis of ECG for another patient.")
    try:
        t, v = exception(t, v)
    except ValueError:
        t = [float(val) for val in t]
        v = [float(val) for val in v]
        t = np.array(t)
        v = np.array(v)
        nans, x = nan_helper(t)
        nans1, x1 = nan_helper(v)
        t[nans] = np.interp(x(nans), x(~nans), t[~nans])
        v[nans1] = np.interp(x1(nans1), x1(~nans1), v[~nans1])
        t = list(t)
        v = list(v)
        logging.error("DATA HAS NaN VALUES! Interpolating...")
    except TypeError:
        t.remove("bad data")
        v.remove("bad data")
        t = [float(val) for val in t]
        v = [float(val) for val in v]
        logging.error("BAD DATA IN SIGNAL! Removing...")
    except SyntaxError:
        t = filter(None, t)
        v = filter(None, v)
        t = [float(val) for val in t]
        v = [float(val) for val in v]
        logging.error("DATA HAS SPACES IN IT! Removing...")
    except OverflowError:
        from warnings import warn
        import sys
        warn("Voltage is too high, stop measuring.")
        logging.warning("VOLTAGE TOO HIGH! SYSTEM OFF!")
        sys.exit()
    dur = duration(t)
    logging.info("Calculating signal duration.")
    max_v, min_v = voltage_extremes(v)
    logging.info("Calculating max and min voltages.")
    print(t)
    print(v)
    print(str(dur)+" seconds")
    print("Minimum voltage: " + str(min_v) +
          "V. Maximum voltage: " + str(max_v) + "V.")
    number_of_beats = num_beats(t, v)
    logging.info("Calculating number of beats.")
    mean_hr_bpm = bpm(dur, number_of_beats)
    logging.info("Calculating mean bpm.")
    peaks_array = beats(t, v)
    logging.info("Outputting time array of beats")
    patient = create_patient(dur, max_v, min_v, number_of_beats,
                             mean_hr_bpm, peaks_array)
    logging.info("Creating patient dictionary.")
    save_patient(patient)
    logging.info("Saving patient dictionary to JSON.")
    print("The number of detected beats is: " +
          str(number_of_beats) + " beats.")
    print("The mean heart rate bpm is " + str(mean_hr_bpm) +
          " beats per minute.")
    print(peaks_array)


if __name__ == "__main__":
    runcode()
