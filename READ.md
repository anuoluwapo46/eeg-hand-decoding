# Left versus right hand decoding from EEG

Author: Kehinde Ayomide Emmanuels

## Summary

This project tries to tell from scalp EEG whether a person is moving, or imagining moving, their left or right hand. I trained a separate model for each person using band-pass filtering, Common Spatial Patterns (CSP) and linear discriminant analysis (LDA). Accuracy depended heavily on the person, from below chance to near perfect. Averaged over groups of ten people, it ranged from 0.50 to 0.64 depending on the setup, with a chance level of about 0.5. Settings that looked better on the first ten people did not carry over to a fresh group of ten, and I report both.

## Data

The PhysioNet EEG Motor Movement/Imagery dataset: 109 volunteers, 64 electrodes, sampled at 160 Hz. I used the first 20 people.

- Runs 4, 8 and 12: imagining opening and closing the left or right fist
- Runs 3, 7 and 11: actually opening and closing the left or right fist

Each person has about 45 trials per set of three runs.

## Method

1. Band-pass filter the signal between 8 and 30 Hz, the range where movement-related changes in brain rhythms appear.
2. Cut the recording into trials from 1 second before to 4 seconds after each cue.
3. Keep the part of each trial from 0.5 to 3.0 seconds after the cue.
4. Extract features with CSP (4 components), which finds combinations of electrodes that best separate left from right trials.
5. Classify with LDA.
6. Evaluate with 10 random splits per person (80% of trials for training, 20% for testing), using the same random seed every time. Accuracy is the mean over the 10 splits.

Models are trained and tested within each person. Nothing is shared between people.

## Results

Average accuracy per group of ten people. Chance level is about 0.5.

| Setup | People 1 to 10 | People 11 to 20 |
|---|---|---|
| Imagery runs, 1 to 2 s window | 0.55 | not run |
| Imagery runs, 0.5 to 3.0 s window | 0.61 | 0.56 |
| Imagery runs, 0.5 to 4.0 s window | 0.57 | not run |
| Imagery runs, CSP regularization added | 0.56 | not run |
| Imagery runs, 21 motor-area electrodes only | 0.63 | 0.50 |
| Real movement runs | 0.62 | not run |
| Imagery and real movement pooled | 0.64 | 0.54 |

I tried the settings above on people 1 to 10. People 11 to 20 were kept as a fresh test group and were only used to check the settings that looked best.

Per-person accuracy for the pooled setup:

| Person | Accuracy | Person | Accuracy |
|---|---|---|---|
| 1 | 0.69 | 11 | 0.61 |
| 2 | 0.71 | 12 | 0.39 |
| 3 | 0.61 | 13 | 0.53 |
| 4 | 0.60 | 14 | 0.59 |
| 5 | 0.48 | 15 | 0.52 |
| 6 | 0.49 | 16 | 0.54 |
| 7 | 0.98 | 17 | 0.52 |
| 8 | 0.64 | 18 | 0.39 |
| 9 | 0.39 | 19 | 0.81 |
| 10 | 0.77 | 20 | 0.47 |

The median is about 0.62 for people 1 to 10 and about 0.53 for people 11 to 20.

## What the results show

- The method works well for a few people and barely at all for many. Person 7 scored 0.96 to 0.99 in every setup, while about half of the people stayed near 0.5.
- The difference between the two groups of ten (roughly ten points) was larger than the effect of any setting I changed. Which people are in the sample mattered more than the choice of window, electrodes or runs.
- Restricting the model to motor-area electrodes looked helpful on people 1 to 10 (0.63) and gave chance-level accuracy on people 11 to 20 (0.50). Pooling imagery and real movement raised the typical score on people 1 to 10 but gave no gain on people 11 to 20.
- Scores below 0.5, such as 0.39, are what noise looks like with about 45 trials per person, not evidence that the model learned the opposite pattern.

## Limits

- Only 20 of the 109 people were used, with about 45 to 90 trials each. Each test split holds roughly 9 to 18 trials, so single-person scores are noisy.
- Test trials come from the same recordings as the training trials, drawn at random. Trials recorded close in time can resemble each other, so scores may be somewhat optimistic. Training on some runs and testing on a separate run would be stricter.
- I tried several settings on people 1 to 10, so the best numbers there are partly the result of choosing among options. The fresh group is the more trustworthy estimate.
- The models are trained per person. I did not test whether a model trained on some people works on someone new.

## Next steps

- Evaluate by holding out whole runs, and then by holding out whole people.
- Use more people from the dataset, so group averages are more stable.
- Try several frequency bands instead of one broad band.
- Try a neural network designed for EEG, such as EEGNet, with the same evaluation.

## How to run

```
pip install mne scikit-learn matplotlib
python all_subjects.py
```

The script downloads the data from PhysioNet on the first run. To change the setup, edit the `runs` list, the `range(...)` in the final loop, and the crop times in `all_subjects.py`.