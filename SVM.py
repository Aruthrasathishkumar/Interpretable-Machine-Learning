
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

# Apply SVM
from sklearn.svm import SVC

clf = SVC(C=1.0, # The smoothing parameter. Smaller values specify stronger regularization. If you have a lot of noisy observations you should decrease it.
                      kernel='rbf', # 'rbf', 'linear', 'poly', 'sigmoid'
                      degree=3) # Degree of the polynomial kernel function ('poly').

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

c_values = np.arange(0.5,2.5, 0.5)


model_settings = []

for degree in [2, 3, 4]:
    for c in c_values:
        model_settings.append(('poly', degree, c))

for c in c_values:
    model_settings.append(('linear', '', c))
    model_settings.append(('sigmoid', '', c))
    model_settings.append(('rbf', '', c))

# Store report rows
results = []

# Store graph lines
rbf_accuracies = []
linear_accuracies = []
sigmoid_accuracies = []
poly2_accuracies = []
poly3_accuracies = []
poly4_accuracies = []

for kernel_name, degree_value, c_value in model_settings:
    print('Kernel:', kernel_name, 'Degree:', degree_value, 'C:', c_value)

    if kernel_name == 'poly':
        clf = SVC(C=c_value, kernel=kernel_name, degree=degree_value)
    else:
        clf = SVC(C=c_value, kernel=kernel_name)

    accuracy = round(cross_val_score(clf, df_feat, df['target'], cv=10, scoring='accuracy').mean(), 3)
    print('10-fold cross-validation accuracy is:', accuracy)
    precision = round(cross_val_score(clf, df_feat, df['target'], cv=10, scoring='precision').mean(), 3)
    print('10-fold cross-validation accuracy is:', precision)
    recall = round(cross_val_score(clf, df_feat, df['target'], cv=10, scoring='recall').mean(), 3)
    print('10-fold cross-validation accuracy is:', recall)
    f1score = round(cross_val_score(clf, df_feat, df['target'], cv=10, scoring='f1').mean(), 3)
    print('10-fold cross-validation accuracy is:', f1score)

    cross_validation_accuracies.append(accuracy)
    cross_validation_precisions.append(precision)
    cross_validation_recalls.append(recall)
    cross_validation_f1scores.append(f1score)

    if kernel_name == 'rbf':
        rbf_accuracies.append(accuracy)
    elif kernel_name == 'linear':
        linear_accuracies.append(accuracy)
    elif kernel_name == 'sigmoid':
        sigmoid_accuracies.append(accuracy)
    elif kernel_name == 'poly' and degree_value == 2:
        poly2_accuracies.append(accuracy)
    elif kernel_name == 'poly' and degree_value == 3:
        poly3_accuracies.append(accuracy)
    elif kernel_name == 'poly' and degree_value == 4:
        poly4_accuracies.append(accuracy)

# One graph with six lines
plt.figure(figsize=(10, 6))
plt.plot(c_values, rbf_accuracies, marker='o', label='RBF Kernel', linestyle='dashed')
plt.plot(c_values, linear_accuracies, marker='o', label='Linear Kernel', linestyle='dashed')
plt.plot(c_values, sigmoid_accuracies, marker='o', label='Sigmoid Kernel', linestyle='dashed')
plt.plot(c_values, poly2_accuracies, marker='o', label='Poly Kernel Degree 2', linestyle='dashed')
plt.plot(c_values, poly3_accuracies, marker='o', label='Poly Kernel Degree 3', linestyle='dashed')
plt.plot(c_values, poly4_accuracies, marker='o', label='Poly Kernel Degree 4', linestyle='dashed')

plt.title('SVM Accuracy VS C value for different kernels')
plt.xlabel('C value')
plt.ylabel('Accuracy')
plt.legend()
plt.show()