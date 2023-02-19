import cv2
import os
import random
import pandas as pd
import numpy as np
import ntpath
from tqdm import tqdm
import joblib

from pgm import write_pgm


def path_leaf(path, keep_extension=True):
    if keep_extension:
        head, tail = ntpath.split(path)
        return tail or ntpath.basename(head)
    else:
        return os.path.splitext(path)[0]


def _task(i, pgm_filename):
    keystroke_matrix = reshape_cmu_row(i)
    data = create_combined_matrix(keystroke_matrix, pgm_filename)
    write_pgm(
        data,
        os.path.join(os.getcwd(), "imgs", pgm_filename[:-4] + "_" + str(i) + ".pgm"),
    )


def generate_combined_image(pgm_filename, parallelize=False):
    df = get_data()
    row_count = df.shape[0]
    if parallelize:
        pgm_filename = pgm_filename[:]
        joblib.Parallel(n_jobs=-1)(
            joblib.delayed(_task)(i, pgm_filename) for i in tqdm(range(row_count))
        )
    else:
        keystroke_matrix = reshape_cmu_row(1)
        data = create_combined_matrix(keystroke_matrix, pgm_filename)
        print(data)
        print(data.shape)
        input("Data")


def get_data():
    df = pd.read_csv(os.path.join(os.getcwd(), "cmu", "cmu_data.csv"))
    # drop all subject and session information from the dataframe
    df = df.drop("subject", axis=1)
    df = df.drop("sessionIndex", axis=1)
    df = df.drop("rep", axis=1)
    return df


def verify_all_pgm_matrix_dimensions():
    pgms = os.listdir(os.path.join(os.getcwd(), "orl"))
    for pgm in pgms:
        matrix = cv2.imread(os.path.join(os.getcwd(), "orl", pgm))
        assert matrix.shape[0] == 112
        assert matrix.shape[1] == 92


def select_random_image(just_filename: bool = False):
    pgms = os.listdir(os.path.join(os.getcwd(), "orl"))
    index = random.randrange(0, len(pgms))
    pgm = pgms[index]
    if just_filename == False:
        return os.path.join(os.getcwd(), "orl", pgm)
    else:
        return path_leaf(os.path.join(os.getcwd(), "orl", pgm))


# TODO: Apparently there are outlier users in the CMU dataset, check the paper for which they are
def reshape_cmu_row(row_idx):
    assert row_idx >= 0
    df = get_data()
    assert row_idx <= df.shape[0]
    row = df.iloc[[row_idx]].to_numpy()[0]
    row = np.pad(row, (0, 112 * 92 - 31), mode="constant")
    # print(row)
    reshaped_row = row.reshape(112, 92)
    return reshaped_row


def create_combined_matrix(keystroke_matrix, pgm_path):
    pgm_matrix = cv2.imread(os.path.join(os.getcwd(), "orl", pgm_path))
    squeezed = pgm_matrix[:, :, 0]
    combined = np.hstack((squeezed, keystroke_matrix))
    assert combined.shape[0] == 112
    assert combined.shape[1] == 184
    return combined


if __name__ == "__main__":
    verify_all_pgm_matrix_dimensions()
    # keystroke_matrix = reshape_cmu_row(1)
    pgm_path = select_random_image(just_filename=True)
    # print(create_combined_matrix(keystroke_matrix, pgm_path).shape)
    generate_combined_image(pgm_path, True)
