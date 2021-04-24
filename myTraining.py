import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression as lg
import pickle

def dataSplit(data, ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_setSize = int(len(data) * ratio)
    test_indices = shuffled[:test_setSize]
    train_indices = shuffled[test_setSize:]
    return data.iloc[train_indices], data.iloc[test_indices]


if __name__ == "__main__":
    df = pd.read_csv('data.csv')

    train, test = dataSplit(df, 0.2)

    x_train = train[['fever', 'bodyPain', 'age', 'runnyNose', 'diffBreath']].to_numpy()
    x_test = test[['fever', 'bodyPain', 'age', 'runnyNose', 'diffBreath']].to_numpy()

    y_train = train[['infectionProb']].to_numpy().reshape(2060 ,)
    y_test = test[['infectionProb']].to_numpy().reshape(515 ,)

    clf = lg()
    clf.fit(x_train, y_train)

    file = open('model.pkl', 'wb')
    pickle.dump(clf, file)

    input = [98, 0, 13, 0, 1]
    inf = clf.predict_proba([input])[0][1]

    file.close()