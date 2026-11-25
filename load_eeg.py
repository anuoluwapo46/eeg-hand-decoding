import mne
from mne.datasets import eegbci

# Subject 1, runs 4, 8 and 12 are the tasks where the person
# imagines moving the left hand or the right hand
subject = 1
runs = [4, 8, 12]

# Download the data files (this only happens the first time)
file_paths = eegbci.load_data(subject, runs)

# Read each file into MNE, then join them into one long recording
recordings = [mne.io.read_raw_edf(path, preload=True) for path in file_paths]
raw = mne.concatenate_raws(recordings)

# Tidy up the channel names so MNE understands them
eegbci.standardize(raw)

# Print a summary of the recording
print(raw.info)

# Show the first 10 seconds of the brain signal
#raw.plot(duration=10, n_channels=15, block=True)

# ---------- Step 2: filter the signal ----------
# Keep only the frequencies between 8 and 30 Hz.
# This is the range where brain activity changes when you imagine moving a hand.
raw.filter(8.0, 30.0)

# Look at the signal again after filtering
#raw.plot(duration=10, n_channels=15, block=True)

# ---------- Step 3: find the moments where the person imagined a movement ----------
# The recording has labels (annotations) like T0, T1 and T2.
# T1 means "imagine the left hand", T2 means "imagine the right hand".
events, _ = mne.events_from_annotations(raw, event_id=dict(T1=2, T2=3))

# ---------- Step 4: cut the recording into short trials (epochs) ----------
# Each trial starts 1 second before the cue and ends 4 seconds after it.
channels = mne.pick_types(raw.info, eeg=True, exclude="bads")

epochs = mne.Epochs(
    raw,
    events,
    event_id=dict(left=2, right=3),
    tmin=-1.0,
    tmax=4.0,
    picks=channels,
    baseline=None,
    preload=True,
)

# Print a summary of the trials
print(epochs)
print("Left trials:", len(epochs["left"]))
print("Right trials:", len(epochs["right"]))

import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline

# ---------- Step 5: prepare the data for the model ----------
# Each trial has a label: 2 means left hand, 3 means right hand
labels = epochs.events[:, -1]

# Use only the part of each trial from 1 to 2 seconds after the cue,
# which is when the brain response to the imagined movement is clearest
epochs_train = epochs.copy().crop(tmin=1.0, tmax=2.0)
data = epochs_train.get_data()

# ---------- Step 6: build the model ----------
# CSP finds the electrode patterns that best separate left from right
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

# LDA draws the dividing line between the two groups
lda = LinearDiscriminantAnalysis()

# Chain them together: first CSP, then LDA
model = Pipeline([("CSP", csp), ("LDA", lda)])

# ---------- Step 7: test the model honestly ----------
# Split the trials 10 different ways. Each time, 80% are used to train
# and 20% are kept hidden to test, so the model is never tested on
# trials it has already seen.
splits = ShuffleSplit(10, test_size=0.2, random_state=42)
scores = cross_val_score(model, data, labels, cv=splits)

# What accuracy would we get by pure guessing?
share_left = np.mean(labels == 2)
chance = max(share_left, 1 - share_left)

print("Accuracy:", round(scores.mean(), 2))
print("Chance level:", round(chance, 2))