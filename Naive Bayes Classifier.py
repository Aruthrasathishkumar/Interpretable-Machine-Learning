import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

# Read the dataset
df = pd.read_csv('pima-indians-diabetes.csv', index_col=0)
feature_names = df.columns[:-1]

# Standardize the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
scaler.fit(df.drop('target', axis=1))
scaled_features = scaler.transform(df.drop('target', axis=1))
df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])

# Split the data into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    scaled_features,
    df['target'],
    test_size=0.3,
    stratify=df['target'],
    random_state=42
)

# Apply Naive Bayes
from sklearn.naive_bayes import GaussianNB

clf = GaussianNB(priors=None)  # default priors from class frequencies
clf = clf.fit(x_train, y_train)

print('Class priors are: ', clf.class_prior_)

# Predictions
predictions_test = clf.predict(x_test)

# Display confusion matrix
confusion_matrix = metrics.confusion_matrix(y_test, predictions_test, labels=clf.classes_)
confusion_matrix_display = metrics.ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix,
    display_labels=clf.classes_
)
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

prior_values = [(i / 10, 1 - i / 10) for i in range(0, 11)]

for p in prior_values:
    print('Priors are:', p)
    clf = GaussianNB(priors=p)

    accuracy = round(cross_val_score(clf, df_feat, df['target'], cv=10, scoring='accuracy').mean(), 3)
    precision = round(cross_val_score(clf, df_feat, df['target'], cv=10, scoring='precision').mean(), 3)
    recall = round(cross_val_score(clf, df_feat, df['target'], cv=10, scoring='recall').mean(), 3)
    f1score = round(cross_val_score(clf, df_feat, df['target'], cv=10, scoring='f1').mean(), 3)

    cross_validation_accuracies.append(accuracy)
    cross_validation_precisions.append(precision)
    cross_validation_recalls.append(recall)
    cross_validation_f1scores.append(f1score)

    print('10-fold CV Accuracy:', accuracy)
    print('10-fold CV Precision:', precision)
    print('10-fold CV Recall:', recall)
    print('10-fold CV F1-score:', f1score)

prior_labels = [f"{int(p[0]*100)}-{int(p[1]*100)}" for p in prior_values]

# Plot Accuracy vs Prior
plt.figure(figsize=(10, 6))
plt.plot(
    prior_labels,
    cross_validation_accuracies,
    color='blue',
    linestyle='dashed',
    marker='o',
    markerfacecolor='red',
    markersize=10
)
plt.title('Accuracy vs. Prior')
plt.xlabel('Prior')
plt.ylabel('Accuracy')
plt.show()