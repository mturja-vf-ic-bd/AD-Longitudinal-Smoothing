from read_file import read_all_subjects, get_baselines
from t_test import ttest
from grouplasso import GroupLassoClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import pickle


if __name__ == '__main__':
    data_set = get_baselines()
    group = ["1", "2"]
    tt = ttest(data_set, group=group)
    X, y, pairs = tt.get_triplet_data(0.3)
    run = 5
    acc = []
    for r in range(run):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        group_ids = np.zeros(X_train.shape[1], dtype=np.int)
        for c in range(0, len(group_ids), 3):
            group_ids[c:c + 3] = c // 3

        sm = SMOTE()
        X_res, y_res = sm.fit_resample(X_train, y_train)
        model = GroupLassoClassifier(group_ids=group_ids, random_state=42, verbose=False, alpha=1e-1, max_iter=2000)
        model.fit(X_train, y_train)
        print(len(model.coef_[np.nonzero(model.coef_)]) // 3)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_pred, y_test)
        acc.append(accuracy)
        print("Accuracy: ", accuracy)
        print("y_pred: {}\ny_test: {}".format(y_pred, y_test))

    print("Average Accuracy: ", np.mean(acc))
    with open('coeff_' + group[0] + '_' + group[1] + '.pkl', 'wb') as f:
        pickle.dump(model.coef_, f)
    with open('t_pairs_' + group[0] + '_' + group[1] + '.pkl', 'wb') as f:
        pickle.dump(pairs, f)
    print("Finished !!!")

