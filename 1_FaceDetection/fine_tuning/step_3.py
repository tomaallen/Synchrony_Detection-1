import os
import random
import numpy as np
from operator import itemgetter
import pandas as pd

if __name__ == "__main__":

    from utils import get_all_files_recursively_by_ext
    sample_directory = r"F:\\Isabella\yolo_samples"

    samples = get_all_files_recursively_by_ext(sample_directory, "jpg")

    # Get all the jpgs
    non_partitioned_dict = {}
    seen_subjects = []
    for sample in samples:
        subject = sample.split(os.sep)[-2]

        if subject not in seen_subjects:
            seen_subjects.append(subject)
            non_partitioned_dict[subject] = []

        non_partitioned_dict[subject].append(sample)

    # Partition the jpgs indistinctively
    partitioned_dict = {}
    num_trials = len(samples)
    random.Random(0).shuffle(samples)
    num_train_trials = int(num_trials * 0.8)
    train_idx = np.arange(num_train_trials)
    validate_idx = np.arange(num_train_trials, num_trials)

    train_trial = list(itemgetter(*train_idx)(samples))
    validate_trial = list(itemgetter(*validate_idx)(samples))

    train_df = pd.DataFrame(train_trial, index=None)
    validate_df = pd.DataFrame(validate_trial, index=None)

    save_path = os.path.join("yolo_train.txt")
    train_df.to_csv(save_path, sep=",", header=None, index=None)

    save_path = os.path.join("yolo_validate.txt")
    validate_df.to_csv(save_path, sep=",", header=None, index=None)
