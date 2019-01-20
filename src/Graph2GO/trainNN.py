from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
import numpy as np
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
from sklearn.metrics import roc_auc_score

def compute_single_roc(preds, labels):
    score = []
    for i in range(preds.shape[1]):
        fpr, tpr, _ = roc_curve(labels[:,i], preds[:,i])
        roc_auc = auc(fpr, tpr)
        score.append(roc_auc)
    return score

def compute_performance(preds, labels):
    preds = np.round(preds, 2)
    labels = labels.astype(np.int32)
    f_max = 0
    p_max = 0
    r_max = 0
    sp_max = 0
    t_max = 0
    for t in range(1, 100):
        threshold = t / 100.0
        predictions = (preds > threshold).astype(np.int32)
        tp = np.sum(predictions * labels)
        fp = np.sum(predictions) - tp
        fn = np.sum(labels) - tp
        sn = tp / (1.0 * np.sum(labels))
        sp = np.sum((predictions ^ 1) * (labels ^ 1))
        sp /= 1.0 * np.sum(labels ^ 1)
        fpr = 1 - sp
        precision = tp / (1.0 * (tp + fp))
        recall = tp / (1.0 * (tp + fn))
        f = 2 * precision * recall / (precision + recall)
        if f_max < f:
            f_max = f
            p_max = precision
            r_max = recall
            sp_max = sp
            t_max = threshold
    return f_max, p_max, r_max, sp_max, t_max


def train_nn(X_train, Y_train, X_test, Y_test, ontology):
    model = Sequential()
    model.add(Dense(1024, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(Y_train.shape[1],activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=100, batch_size=128, verbose=0)

    y_prob = model.predict(X_test)
    np.save("../../data/predict_" + ontology + ".npy",y_prob)
    np.save("../../data/True_" + ontology + ".npy",Y_test)
    
    scores = compute_single_roc(y_prob,Y_test)

    print("ROC-AUC:",np.nanmean(scores))
    f_max, p_max, r_max, sp_max, t_max = compute_performance(y_prob, Y_test)
    print("F-max:",f_max)
    





    
