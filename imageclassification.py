import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.mixture import GaussianMixture
from hmmlearn.hmm import GaussianHMM
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import metrics

dataset = pd.read_csv('dataset.csv')
X= dataset.iloc[:,0:17]
y= dataset.iloc[:,17]
sc = StandardScaler()
X = sc.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#Random Forest
clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
val_pred = clf.predict(X_test)
print("Accuracy-Random Forest:",metrics.accuracy_score(y_test, val_pred))
print("Precision_RF:", precision_score(y_test, val_pred))
print("Recall_RF:", recall_score(y_test, val_pred))
print("F1 score_RF:", f1_score(y_test, val_pred))
lr_fpr, lr_tpr, _ = roc_curve(y_test, val_pred)
AUC_RandomForest = roc_auc_score(y_test, val_pred)
plt.plot(lr_fpr, lr_tpr, marker='.', label='Random Forest', color = 'red')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

#SVM
clf = svm.SVC(kernel='rbf', C = 1, gamma = 1)
clf.fit(X_train,y_train)
val_pred = clf.predict(X_test)
print("Accuracy-SVM:",metrics.accuracy_score(y_test, val_pred))
print("Precision_SVM:", precision_score(y_test, val_pred))
print("Recall_SVM:", recall_score(y_test, val_pred))
print("F1 score_SVM:", f1_score(y_test, val_pred))
lr_fpr, lr_tpr, _ = roc_curve(y_test, val_pred)
AUC_SVM = roc_auc_score(y_test, val_pred)
plt.plot(lr_fpr, lr_tpr, marker='.', label='SVM', color = 'blue')

clf = svm.SVC(kernel='rbf', C = 1, gamma = 11)
clf.fit(X_train,y_train)
val_pred = clf.predict(X_test)
print("Accuracy-SVM:",metrics.accuracy_score(y_test, val_pred))
print("Precision_SVM:", precision_score(y_test, val_pred))
print("Recall_SVM:", recall_score(y_test, val_pred))
print("F1 score_SVM:", f1_score(y_test, val_pred))
lr_fpr, lr_tpr, _ = roc_curve(y_test, val_pred)
AUC_SVM = roc_auc_score(y_test, val_pred)
plt.plot(lr_fpr, lr_tpr, marker='.', label='LSTM', color = 'magenta')

# KNN
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train,y_train)
val_pred = clf.predict(X_test)
print("Accuracy-KNN:",metrics.accuracy_score(y_test, val_pred))
print("Precision_KNN:", precision_score(y_test, val_pred))
print("Recall_KNN:", recall_score(y_test, val_pred))
print("F1 score_KNN:", f1_score(y_test, val_pred))
lr_fpr, lr_tpr, _ = roc_curve(y_test, val_pred)
AUC_KNN = roc_auc_score(y_test, val_pred)
plt.plot(lr_fpr, lr_tpr, marker='.', label='KNN', color = 'green')

#Adaboost
clf = AdaBoostClassifier(n_estimators=50, learning_rate=1)
clf.fit(X_train,y_train)
val_pred = clf.predict(X_test)
print("Accuracy-Adaboost:",metrics.accuracy_score(y_test, val_pred))
print("Precision_AB:", precision_score(y_test, val_pred))
print("Recall_AB:", recall_score(y_test, val_pred))
print("F1 score_AB:", f1_score(y_test, val_pred))
lr_fpr, lr_tpr, _ = roc_curve(y_test, val_pred)
AUC_Adaboost = roc_auc_score(y_test, val_pred)
plt.plot(lr_fpr, lr_tpr, marker='.', label='Adaboost', color = 'orange')

#Naive bayes
clf = GaussianNB()
clf.fit(X_train,y_train)
val_pred = clf.predict(X_test)
print("Accuracy-Naive Bayes:",metrics.accuracy_score(y_test, val_pred))
print("Precision_NB:", precision_score(y_test, val_pred))
print("Recall_NB:", recall_score(y_test, val_pred))
print("F1 score_NB:", f1_score(y_test, val_pred))
lr_fpr, lr_tpr, _ = roc_curve(y_test, val_pred)
AUC_NaiveBayes = roc_auc_score(y_test, val_pred)
plt.plot(lr_fpr, lr_tpr, marker='.', label='Naive bayes', color = 'black')



# print("Random Forest: ", AUC_RandomForest)
# print("SVM: ", AUC_SVM)
# print("KNN: ", AUC_KNN)
# print("Adaboost: ", AUC_Adaboost)
# print("Naive Bayes: ", AUC_NaiveBayes)


# dataset = pd.read_csv('new_dataset.csv')
# X= dataset.iloc[:,0:17]
# # X= X.reshape((X.shape[0], X.shape[1], 1))
# # y= dataset.iloc[:,17]
# y= pd.get_dummies(dataset.iloc[:,17]).values
# sc = StandardScaler()
# X = sc.fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# dfobj = pd.DataFrame(y_test)
# ytest = pd.Series(dfobj.columns[np.where(dfobj!=0)[1]])
# xtest = X_test
# X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
# X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
# # print(X_train.shape,y_train.shape)
# # print(X_test.shape,y_test.shape)


# model = Sequential()
# model.add(LSTM(32, dropout=0.1, recurrent_dropout=0.1,return_sequences=True,input_shape = (17,1)))
# model.add(LSTM(32, dropout=0.1, recurrent_dropout=0.1))
# model.add(Dense(2, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# epochs =200
# batch_size = 20

# history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=0.2)

# val_pred = model.predict_classes(X_test)
# y_pred = model.predict_classes(X_test)

# lr_fpr, lr_tpr, _ = roc_curve(ytest, y_pred)
# plt.plot(lr_fpr, lr_tpr, marker='.', label='LSTM', color = 'magenta')


plt.legend()
plt.show()
# y_pred=clf.predict(X_test)
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))



# Accuracy-Random Forest: 0.9130434782608695
# Accuracy-SVM: 0.9619565217391305
# Accuracy-KNN: 0.9239130434782609
# Accuracy-Adaboost: 0.875
# Accuracy-Naive Bayes: 0.842391304347826

# Random Forest:  0.8577418853859295
# SVM:  0.9534865662369933
# KNN:  0.8720298182947662
# Adaboost:  0.8252057772946109
# Naive Bayes:  0.7054666873738158
