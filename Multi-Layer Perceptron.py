import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

df = pd.read_csv('pima-indians-diabetes.csv', index_col=0)
feature_names = df.columns[:-1]
# print(df.head())

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

from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(random_state=1, # Pass an int for reproducible results across multiple function calls.
                    hidden_layer_sizes=(20, 20), # tuple, length = n_layers - 2, default=(100,). The ith element represents the number of neurons in the ith hidden layer.
                    activation='relu', # {'identity', 'logistic', 'tanh', 'relu'}, default='relu'. Activation function for the hidden layer.
                    solver='adam', # {'lbfgs', 'sgd', 'adam'}, default='adam'. The solver for weight optimization.
                    alpha=0.00001, # L2 penalty (regularization term) parameter.
                    batch_size='auto', # int, default='auto'. Size of minibatches for stochastic optimizers. When set to 'auto', batch_size=min(200, n_samples).
                    learning_rate='adaptive', # {'constant', 'invscaling', 'adaptive'}, default='constant'. Learning rate schedule for weight updates.
                    # 'constant' is a constant learning rate given by 'learning_rate_init'.
                    # 'invscaling' gradually decreases the learning rate at each time step 't' using an inverse scaling exponent of 'power_t'. effective_learning_rate = learning_rate_init / pow(t, power_t)
                    # 'adaptive' keeps the learning rate constant to 'learning_rate_init' as long as training loss keeps decreasing. Each time two consecutive epochs fail to decrease training loss by at least tol, or fail to increase validation score by at least tol if 'early_stopping' is on, the current learning rate is divided by 5.
                    learning_rate_init=0.001, # The initial learning rate used. It controls the step-size in updating the weights.
                    max_iter=1000, # Maximum number of iterations. The solver iterates until convergence (determined by 'tol') or this number of iterations.
                    shuffle=True, # Whether to shuffle samples in each iteration.
                    tol=0.0001, # default=1e-4. Tolerance for the optimization.
                    # When the loss or score is not improving by at least tol for n_iter_no_change consecutive iterations, convergence is considered to be reached and training stops.
                    early_stopping=False, # Whether to use early stopping to terminate training when validation score is not improving.
                    # If set to true, it will automatically set aside 10% of training data as validation and terminate training when validation score is not improving by at least tol for n_iter_no_change consecutive epochs.
                    n_iter_no_change=10) # Maximum number of epochs to not meet tol improvement. Only effective when solver='sgd' or 'adam'.

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

activation_functions = ['relu', 'identity', 'logistic', 'tanh']
hidden_layer_sizes_list = [(10,), (20,), (10, 10), (20, 20)]

relu_accuracies = []
identity_accuracies = []
logistic_accuracies = []
tanh_accuracies = []

for activation_name in activation_functions:
    for hidden_size in hidden_layer_sizes_list:
        # Pipeline ensures scaling is done separately inside each CV fold
        clf = MLPClassifier(random_state=1,  # Pass an int for reproducible results across multiple function calls.
                            hidden_layer_sizes=hidden_size,
                            # tuple, length = n_layers - 2, default=(100,). The ith element represents the number of neurons in the ith hidden layer.
                            activation=activation_name,
                            # {'identity', 'logistic', 'tanh', 'relu'}, default='relu'. Activation function for the hidden layer.
                            solver='adam',
                            # {'lbfgs', 'sgd', 'adam'}, default='adam'. The solver for weight optimization.
                            alpha=0.00001,  # L2 penalty (regularization term) parameter.
                            batch_size='auto',
                            # int, default='auto'. Size of minibatches for stochastic optimizers. When set to 'auto', batch_size=min(200, n_samples).
                            learning_rate='adaptive',
                            # {'constant', 'invscaling', 'adaptive'}, default='constant'. Learning rate schedule for weight updates.
                            # 'constant' is a constant learning rate given by 'learning_rate_init'.
                            # 'invscaling' gradually decreases the learning rate at each time step 't' using an inverse scaling exponent of 'power_t'. effective_learning_rate = learning_rate_init / pow(t, power_t)
                            # 'adaptive' keeps the learning rate constant to 'learning_rate_init' as long as training loss keeps decreasing. Each time two consecutive epochs fail to decrease training loss by at least tol, or fail to increase validation score by at least tol if 'early_stopping' is on, the current learning rate is divided by 5.
                            learning_rate_init=0.001,
                            # The initial learning rate used. It controls the step-size in updating the weights.
                            max_iter=1000,
                            # Maximum number of iterations. The solver iterates until convergence (determined by 'tol') or this number of iterations.
                            shuffle=True,  # Whether to shuffle samples in each iteration.
                            tol=0.0001,  # default=1e-4. Tolerance for the optimization.
                            # When the loss or score is not improving by at least tol for n_iter_no_change consecutive iterations, convergence is considered to be reached and training stops.
                            early_stopping=False,
                            # Whether to use early stopping to terminate training when validation score is not improving.
                            # If set to true, it will automatically set aside 10% of training data as validation and terminate training when validation score is not improving by at least tol for n_iter_no_change consecutive epochs.
                            n_iter_no_change=10)  # Maximum number of epochs to not meet tol improvement. Only effective when solver='sgd' or 'adam'.

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

        if activation_name == 'relu':
            relu_accuracies.append(accuracy)
        elif activation_name == 'identity':
            identity_accuracies.append(accuracy)
        elif activation_name == 'logistic':
            logistic_accuracies.append(accuracy)
        elif activation_name == 'tanh':
            tanh_accuracies.append(accuracy)

# Main Graph for overall accuracy versus hidden layer size
labels = ['(10)', '(20)', '(10, 10)', '(20, 20)']

plt.figure(figsize=(10, 6))
plt.plot(labels, relu_accuracies, marker='o', label='relu', linestyle='dashed')
plt.plot(labels, identity_accuracies, marker='o', label='identity', linestyle='dashed')
plt.plot(labels, logistic_accuracies, marker='o', label='logistic', linestyle='dashed')
plt.plot(labels, tanh_accuracies, marker='o', label='tanh', linestyle='dashed')

plt.title('MLP Overall Accuracy vs Hidden Layer Size')
plt.xlabel('Hidden Layer Size')
plt.ylabel('Overall Accuracy')
plt.legend()
plt.show()