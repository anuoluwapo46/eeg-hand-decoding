import numpy as np
import mne
from mne.datasets import eegbci
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline

# Print less text while the code runs
mne.set_log_level("WARNING")


def score_one_subject(subject):
    # Runs 4, 8 and 12 are the left versus right hand imagery tasks
    #runs = [3, 7, 11]
    runs = [3, 4, 7, 8, 11, 12]

    # Download and load this person's recordings
    file_paths = eegbci.load_data(subject, runs)
    recordings = [mne.io.read_raw_edf(path, preload=True) for path in file_paths]
    raw = mne.concatenate_raws(recordings)
    eegbci.standardize(raw)
    # Keep only the electrodes over the motor area of the brain
    motor_channels = [
        "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
        "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
        "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
    ]
    #raw.pick(motor_channels)

    # Keep only the 8 to 30 Hz band
    raw.filter(8.0, 30.0)

    # Find the cues and cut the recording into trials
    events, _ = mne.events_from_annotations(raw, event_id=dict(T1=2, T2=3))
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

    # Labels and the useful 1 to 2 second window
    labels = epochs.events[:, -1]
    #data = epochs.copy().crop(tmin=0.5, tmax=4.0).get_data()
    data = epochs.copy().crop(tmin=0.5, tmax=3.0).get_data()
    #data = epochs.copy().crop(tmin=1.0, tmax=2.0).get_data()

    # The same model as before: CSP followed by LDA
    model = Pipeline([
        ("CSP", CSP(n_components=4, reg=None, log=True, norm_trace=False)),
        ("LDA", LinearDiscriminantAnalysis()),
    ])

    # Train and test 10 times, always on separate trials
    splits = ShuffleSplit(10, test_size=0.2, random_state=42)
    scores = cross_val_score(model, data, labels, cv=splits)
    return scores.mean()


# Run the same steps for subjects 1 to 10
all_scores = []
for subject in range(11, 21):
    accuracy = score_one_subject(subject)
    all_scores.append(accuracy)
    print("Subject", subject, "accuracy:", round(accuracy, 2))

print("Average accuracy:", round(np.mean(all_scores), 2))