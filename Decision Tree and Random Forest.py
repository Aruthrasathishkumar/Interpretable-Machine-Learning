
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

# Apply Decision Tree
from sklearn import tree

clf = tree.DecisionTreeClassifier(max_depth=None, # You can set a maximum depth for the tree to prevent overfitting
                                  min_samples_split=7, # The minimum number of samples required to split an internal node.
                                  min_samples_leaf=2, random_state=42) # A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right.


# Apply random forest
from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(max_depth=None, # You can set a maximum depth for the tree to prevent overfitting
#                                           min_samples_split=7, # The minimum number of samples required to split an internal node.
#                                           min_samples_leaf=2, #A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right.
#                                           n_estimators=100, random_state=42)  # The number of trees in the forest

clf = clf.fit(x_train, y_train)

# Predictions
predictions_test = clf.predict(x_test)

# Display confusion matrix
confusion_matrix = metrics.confusion_matrix(y_test, predictions_test, labels=clf.classes_)
confusion_matrix_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=clf.classes_)
confusion_matrix_display.plot()
from matplotlib import pyplot as plt
# plt.show()

# Report Overall Accuracy, precision, recall, F1-score
class_names = list(map(str, clf.classes_))
print(metrics.classification_report(
    y_true=y_test,
    y_pred=predictions_test,
    target_names=class_names,
    zero_division=0
))

# Produce visualizations of the tree graph.
class_names = list(map(str, clf.classes_))
plt.figure(figsize=(16, 8))
tree.plot_tree(
    decision_tree=clf,
    max_depth=3,
    feature_names=feature_names,
    class_names=class_names,
    filled=True
)
plt.show()

from sklearn.model_selection import cross_val_score

cross_validation_accuracies = []
cross_validation_precisions = []
cross_validation_recalls = []
cross_validation_f1scores = []

min_split_values = np.arange(5,25, 5)
min_leaf_values = np.arange(3,19, 4)


# decision tree
dt_5_accuracies = []
dt_10_accuracies = []
dt_15_accuracies = []
dt_20_accuracies = []

for split_value in min_split_values:
    for leaf_value in min_leaf_values:
        clf = tree.DecisionTreeClassifier(
            max_depth=None,
            min_samples_split=split_value,
            min_samples_leaf=leaf_value,
            random_state=42
        )

        accuracy = round(cross_val_score(clf, df_feat, df['target'], cv=10, scoring='accuracy').mean(), 3)
        precision = round(cross_val_score(clf, df_feat, df['target'], cv=10, scoring='precision').mean(), 3)
        recall = round(cross_val_score(clf, df_feat, df['target'], cv=10, scoring='recall').mean(), 3)
        f1score = round(cross_val_score(clf, df_feat, df['target'], cv=10, scoring='f1').mean(), 3)

        print('Classifier: Decision Tree', 'min_samples_split:', split_value, 'min_samples_leaf:', leaf_value)
        print('10-fold cross-validation accuracy is:', accuracy)
        print('10-fold cross-validation precision is:', precision)
        print('10-fold cross-validation recall is:', recall)
        print('10-fold cross-validation f1-score is:', f1score)

        cross_validation_accuracies.append(accuracy)
        cross_validation_precisions.append(precision)
        cross_validation_recalls.append(recall)
        cross_validation_f1scores.append(f1score)

        if split_value == 5:
            dt_5_accuracies.append(accuracy)
        elif split_value == 10:
            dt_10_accuracies.append(accuracy)
        elif split_value == 15:
            dt_15_accuracies.append(accuracy)
        elif split_value == 20:
            dt_20_accuracies.append(accuracy)

# decision tree graph
plt.figure(figsize=(10, 6))
plt.plot(min_leaf_values, dt_5_accuracies, marker='o', label='min_samples_split = 5', linestyle='dashed')
plt.plot(min_leaf_values, dt_10_accuracies, marker='o', label='min_samples_split = 10', linestyle='dashed')
plt.plot(min_leaf_values, dt_15_accuracies, marker='o', label='min_samples_split = 15', linestyle='dashed')
plt.plot(min_leaf_values, dt_20_accuracies, marker='o', label='min_samples_split = 20', linestyle='dashed')
plt.title('Decision Tree Accuracy VS min_samples_leaf')
plt.xlabel('min_samples_leaf')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# random forest
rf_5_accuracies = []
rf_10_accuracies = []
rf_15_accuracies = []
rf_20_accuracies = []

for split_value in min_split_values:
    for leaf_value in min_leaf_values:
        clf = RandomForestClassifier(
            max_depth=None,
            min_samples_split=split_value,
            min_samples_leaf=leaf_value,
            n_estimators=100,
            random_state=42
        )

        accuracy = round(cross_val_score(clf, df_feat, df['target'], cv=10, scoring='accuracy').mean(), 3)
        precision = round(cross_val_score(clf, df_feat, df['target'], cv=10, scoring='precision').mean(), 3)
        recall = round(cross_val_score(clf, df_feat, df['target'], cv=10, scoring='recall').mean(), 3)
        f1score = round(cross_val_score(clf, df_feat, df['target'], cv=10, scoring='f1').mean(), 3)

        print('Classifier: Random Forest', 'min_samples_split:', split_value, 'min_samples_leaf:', leaf_value)
        print('10-fold cross-validation accuracy is:', accuracy)
        print('10-fold cross-validation precision is:', precision)
        print('10-fold cross-validation recall is:', recall)
        print('10-fold cross-validation f1-score is:', f1score)

        cross_validation_accuracies.append(accuracy)
        cross_validation_precisions.append(precision)
        cross_validation_recalls.append(recall)
        cross_validation_f1scores.append(f1score)

        if split_value == 5:
            rf_5_accuracies.append(accuracy)
        elif split_value == 10:
            rf_10_accuracies.append(accuracy)
        elif split_value == 15:
            rf_15_accuracies.append(accuracy)
        elif split_value == 20:
            rf_20_accuracies.append(accuracy)

# random forest graph
plt.figure(figsize=(10, 6))
plt.plot(min_leaf_values, rf_5_accuracies, marker='o', label='min_samples_split = 5', linestyle='dashed')
plt.plot(min_leaf_values, rf_10_accuracies, marker='o', label='min_samples_split = 10', linestyle='dashed')
plt.plot(min_leaf_values, rf_15_accuracies, marker='o', label='min_samples_split = 15', linestyle='dashed')
plt.plot(min_leaf_values, rf_20_accuracies, marker='o', label='min_samples_split = 20', linestyle='dashed')
plt.title('Random Forest Accuracy VS min_samples_leaf')
plt.xlabel('min_samples_leaf')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


