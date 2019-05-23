from read_file import read_all_subjects, get_baselines
from t_test import ttest
from grouplasso import GroupLassoClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE


if __name__ == '__main__':
    data_set = get_baselines()
    tt = ttest(data_set, group=["1", "3"])
    X, y = tt.get_triplet_data(0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    group_ids = np.zeros(X_train.shape[1], dtype=np.int)
    for c in range(0, len(group_ids), 3):
        group_ids[c:c + 3] = c // 3


    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    model = GroupLassoClassifier(group_ids=group_ids, random_state=42, verbose=False, alpha=1e-1, max_iter=2000)
    model.fit(X_train, y_train)
    print(len(model.coef_[np.nonzero(model.coef_)]) // 3)
    y_pred = model.predict(X_test)
    print(accuracy_score(y_pred, y_test))
    print("y_pred: {}\ny_test: {}".format(y_pred, y_test))