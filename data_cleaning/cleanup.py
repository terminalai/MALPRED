import pandas as pd
from sklearn.model_selection import StratifiedKFold
import sys
from constants import dtypes
from preprocessing import clean_df

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        filename = "/kaggle/input/microsoft-malware-prediction/train.csv"
    else:
        filename = sys.argv[1]

    train = pd.read_csv(filename, dtype=dtypes)

    train = clean_df(train)

    dtypes = train.dtypes.to_dict()

    CATEGORICAL_FEATURES = []
    NUMERIC_FEATURES = []
    NUM_CATEGORIES = {}
    LABEL = "HasDetections"

    for col, val in dtypes.items():
        if col == "HasDetections":
            continue

        if val.type.__name__ == "CategoricalDtypeType":
            CATEGORICAL_FEATURES.append(col)
            NUM_CATEGORIES[col] = len(train[col].cat.categories)
            train[col] = train[col].cat.codes

        else:
            NUMERIC_FEATURES.append(col)
            train[col] = train[col].astype(float)

    FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    train = train[FEATURES + [LABEL]]

    train.to_csv("data.csv", index=False)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    for i, (train_index, test_index) in enumerate(skf.split(train[FEATURES], train[LABEL])):
        train.iloc[train_index].to_csv(f"fold{i}_train.csv", index=False)
        train.iloc[test_index].to_csv(f"fold{i}_test.csv", index=False)
