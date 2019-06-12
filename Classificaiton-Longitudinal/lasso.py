from read_file import read_all_subjects, get_baselines, get_strat_label
from t_test import ttest
from grouplasso import GroupLassoClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import pickle
from arg import Args
from operator import itemgetter
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

def get_train_test_fold(dataset, ratio=0.2):
    data_set_train = dict()
    data_set_test = dict()
    # zipped_data = list(zip(dataset["node_feature"], dataset["adjacency_matrix"], dataset["dx_label"]))
    # random.shuffle(zipped_data)
    # nf, am, dx = zip(*zipped_data)
    # n = len(nf)
    n_fold = int(1/ratio)
    kf = StratifiedKFold(n_splits=n_fold, shuffle=True)
    train_fold = []
    test_fold = []

    for train_index, test_index in kf.split(data_set["adjacency_matrix"], dataset["dx_label"]):
        data_set_test["node_feature"] = itemgetter(*test_index)(dataset["node_feature"])
        data_set_test["adjacency_matrix"] = itemgetter(*test_index)(data_set["adjacency_matrix"])
        data_set_test["dx_label"] = itemgetter(*test_index)(dataset["dx_label"])

        data_set_train["node_feature"] = itemgetter(*train_index)(dataset["node_feature"])
        data_set_train["adjacency_matrix"] = itemgetter(*train_index)(data_set["adjacency_matrix"])
        data_set_train["dx_label"] = itemgetter(*train_index)(dataset["dx_label"])

        test_fold.append(data_set_test)
        train_fold.append(data_set_train)

    return list(zip(train_fold, test_fold))



if __name__ == '__main__':
    y_strat = get_strat_label()
    data_set = get_baselines(net_dir=Args.NETWORK_DIR, label=y_strat)
    train_test_fold = get_train_test_fold(data_set)
    group = ["2", "3"]
    acc = []
    for train, test in train_test_fold:
        tt = ttest(train, group=group)
        tt_test = ttest(test, group=group)
        X_train, y_train, pairs = tt.get_link_data(0.15)
        print("n_link: ", len(pairs))
        X_test, y_test, _ = tt_test.get_link_data(pairs=pairs)

        sm = SMOTE()
        X_res, y_res = sm.fit_resample(X_train, y_train)

        # group_ids = np.zeros(X_train.shape[1], dtype=np.int)
        # for c in range(0, len(group_ids), 3):
        #     group_ids[c:c + 3] = c // 3
        # group_ids = np.array([i for i in range(0, X_train.shape[1])])
        # model = GroupLassoClassifier(group_ids=group_ids, random_state=42, verbose=False, alpha=1e-1, max_iter=2000)
        # model.fit(X_res, y_res)
        # print(len(model.coef_[np.nonzero(model.coef_)]) // 3)
        model = LogisticRegression(penalty='l1', solver='saga', tol=0.01)
        model.fit(X_res, y_res)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_pred, y_test)
        if accuracy < 0.5:
            accuracy = 1 - accuracy
        acc.append(accuracy)
        print("Accuracy: ", accuracy)
        print("y_pred: {}\ny_test: {}".format(y_pred, y_test))

    print("Average Accuracy: ", np.mean(acc))
    with open('coeff_' + group[0] + '_' + group[1] + '.pkl', 'wb') as f:
        pickle.dump(model.coef_, f)
    with open('t_pairs_' + group[0] + '_' + group[1] + '.pkl', 'wb') as f:
        pickle.dump(pairs, f)
    print("Finished !!!")

