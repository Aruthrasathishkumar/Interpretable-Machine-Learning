
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

df = pd.read_csv('pima-indians-diabetes.csv', index_col=0)
feature_names = df.columns[:-1]
print(df.head())

# Standardize the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
scaler.fit(df.drop('target', axis=1))
scaled_features = scaler.transform(df.drop('target', axis=1))
df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])
print(df_feat.head())


# Split the data into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(scaled_features, df['target'], test_size=0.3, stratify=df['target'], random_state=42)

# Apply Logistic Regression
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(C=2) # Inverse of regularization strength; must be a positive float. Smaller values specify stronger regularization.
clf = clf.fit(x_train, y_train)

# Predictions
predictions_test = clf.predict(x_test)

# Display confusion matrix
confusion_matrix = metrics.confusion_matrix(y_test, predictions_test, labels=clf.classes_)
confusion_matrix_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=clf.classes_)
confusion_matrix_display.plot()
from matplotlib import pyplot as plt
plt.show()

# Report Overall Accuracy, precision, recall, F1-score
class_names = list(map(str, clf.classes_))
print(metrics.classification_report(
    y_true=y_test,
    y_pred=predictions_test,
    target_names=class_names,
    zero_division=0
))


from sklearn.model_selection import cross_val_score

cross_validation_accuracies = []
cross_validation_precisions = []
cross_validation_recalls = []
cross_validation_f1scores = []

c_values = np.arange(0.05, 1.05, 0.05)

for i in c_values:
    print('C is:', round(i, 2))
    clf = LogisticRegression(C=i,max_iter=1000)

    accuracy = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='accuracy').mean()
    cross_validation_accuracies.append(accuracy)
    print('10-fold cross-validation accuracy is:', accuracy)

    precision = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='precision').mean()
    cross_validation_precisions.append(precision)
    print('10-fold cross-validation precision is:', precision)

    recall = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='recall').mean()
    cross_validation_recalls.append(recall)
    print('10-fold cross-validation recall is:', recall)

    f1score = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='f1').mean()
    cross_validation_f1scores.append(f1score)
    print('10-fold cross-validation f1score is:', f1score)

# Create a graph that shows the overall accuracy for different values of the hyperparameter.
plt.figure(figsize=(10, 6))
plt.plot(c_values, cross_validation_accuracies, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.xticks(c_values, [round(i, 2) for i in c_values], rotation=45)
plt.title('Accuracy vs. C- values')
plt.xlabel('C - value')
plt.ylabel('Accuracy')
plt.show()