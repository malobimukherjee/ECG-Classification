import numpy as np
import wfdb
from scipy.signal import butter, filtfilt
from wfdb import processing

def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=360, order=4):
    nyquist = 0.5 * fs  # Nyquist Frequency
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    filtered_signal = filtfilt(b, a, signal, axis=0)  # Apply filter along the 0-axis
    return filtered_signal

# Function to apply FFT to a signal segment
def apply_fft(segment):
    fft_segment = np.fft.fft(segment)
    fft_magnitude = np.abs(fft_segment)  # Magnitude of the FFT
    return fft_magnitude[:len(fft_magnitude) // 2]  # Return only the positive frequencies

# Function to extract beats around the QRS peak with FFT
def extract_beats(signal, qrs_inds, window_before=100, window_after=130, fs=360):
    # Filter the signal
    filtered_signal = bandpass_filter(signal, fs=fs)

    beats = []
    for qrs_index in qrs_inds:
        # Ensure we can safely extract a window around the peak
        if qrs_index >= window_before and qrs_index + window_after < len(filtered_signal):
            beat = filtered_signal[qrs_index - window_before: qrs_index + window_after]
            if len(beat) == window_before + window_after:  # Ensure beat is 360 samples
                beat_channel = beat[:, 0]  # Extract the first channel (if it's a 2D array)

                # Apply FFT to the extracted beat
                #beat_fft = apply_fft(beat_channel)

                # Append the FFT-transformed beat to the list
                beats.append(beat_channel)

    return beats
# Iterate over patient records from 100 to 108
normal_beats = []
abnormal_beats = []

for patient in range(100, 109):
    #print(f"\nProcessing patient record: {patient}")

    # Load the ECG signal and annotations
    signal, fields = wfdb.rdsamp(f'mitdb/{patient}', channels=[0])
    annotations = wfdb.rdann(f'mitdb/{patient}', 'atr')

    # Initialize XQRS for QRS detection
    xqrs = processing.XQRS(sig=signal[:, 0], fs=fields['fs'])
    xqrs.detect()

    #print(f"Total QRS Peaks Detected for patient {patient}: {len(xqrs.qrs_inds)}")

    # Separate QRS peaks into normal and abnormal based on annotations
    normal_qrs_inds = []
    abnormal_qrs_inds = []

    for ann_index, symbol in zip(annotations.sample, annotations.symbol):
        if symbol == 'N':
            if ann_index in xqrs.qrs_inds:  # Ensure the annotation matches detected QRS
                normal_qrs_inds.append(ann_index)
        else:
            if ann_index in xqrs.qrs_inds:  # Ensure the annotation matches detected QRSf
                abnormal_qrs_inds.append(ann_index)

    #print(f"Number of Normal Beats (N): {len(normal_qrs_inds)}")
    #print(f"Number of Abnormal Beats (non-N): {len(abnormal_qrs_inds)}")

    # Extract 2 normal beats
    if len(normal_qrs_inds) >= 50:
        normal_beats.extend(extract_beats(signal, normal_qrs_inds[:50]))

    # Extract 2 abnormal beats
    if len(abnormal_qrs_inds) >= 50:
        abnormal_beats.extend(extract_beats(signal, abnormal_qrs_inds[:50]))

# Convert the lists of beats into NumPy arrays
normal_beats_array = np.array(normal_beats)
abnormal_beats_array = np.array(abnormal_beats)

print(f" Normal beat array shape:{normal_beats_array.shape}")
print(f" abnormal beat array shape:{abnormal_beats_array.shape}")

# Print the number of extracted segments
print(f"Normal segments extracted: {normal_beats_array.shape[0]}")
print(f"Abnormal segments extracted: {abnormal_beats_array.shape[0]}")

print(f"Normal segments: {normal_beats_array}")
print(f"Abnormal segments: {abnormal_beats_array}")
# Interleave abnormal and normal beats: 2 abnormal, 2 normal
combined_beats = []
abnormal_idx, normal_idx = 0, 0


# Loop to alternate 2 abnormal and 2 normal beats
while abnormal_idx < len(abnormal_beats_array) or normal_idx < len(normal_beats_array):
    # Add 2 abnormal beats with labels
    for beat in abnormal_beats_array[abnormal_idx:abnormal_idx + 2]:
        combined_beats.append(np.insert(beat, 0, 0))  # Insert 0 for abnormal
    abnormal_idx += 2
    # Add 2 normal beats with labels
    for beat in normal_beats_array[normal_idx:normal_idx + 2]:
        combined_beats.append(np.insert(beat, 0, 1))  # Insert 1 for normal
    normal_idx += 2

# Convert the combined list into a NumPy array
combined_beats_array = np.array(combined_beats)
# Save the combined array to a CSV file
np.savetxt('C:/Users/DELL/OneDrive/Documents/combined_beats.csv', combined_beats_array, delimiter=',', header='Label,ECG Beats', comments='')