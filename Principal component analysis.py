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

# Apply PCA to reduce data to 2 dimensions for visualization
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
x_train_2 = pca.fit_transform(x_train)
x_test_2 = pca.transform(x_test)

# 1. Graph for Ordinary Least Squares classifier
from sklearn import linear_model

clf_pca = linear_model.LinearRegression()
clf_pca = clf_pca.fit(x_train_2, y_train)

# Predictions
predictions_test = clf_pca.predict(x_test_2)
predictions_test[predictions_test <= 0.5] = 0
predictions_test[predictions_test > 0.5] = 1
predictions_test = predictions_test.astype(int)

# Scatter plot
y_train_2 = y_train.tolist()
plt.figure(figsize=(8, 6))
plt.scatter(x_train_2[:, 0], x_train_2[:, 1], c=y_train_2, s=10, cmap='viridis')

# Create a mesh to plot in
first_dimension_min, first_dimension_max = x_train_2[:, 0].min() - 1, x_train_2[:, 0].max() + 1
second_dimension_min, second_dimension_max = x_train_2[:, 1].min() - 1, x_train_2[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.arange(first_dimension_min, first_dimension_max, 0.02),
    np.arange(second_dimension_min, second_dimension_max, 0.02)
)

Z = clf_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z[Z <= 0.5] = 0
Z[Z > 0.5] = 1
Z = Z.astype(int)
Z = Z.reshape(xx.shape)

plt.contour(xx, yy, Z)
plt.title('Ordinary Least Squares Classifier')
plt.axis('off')
plt.show()


# 2. Graph for Ridge Regression
clf_pca = linear_model.Ridge(alpha=0.5, random_state=0)
clf_pca = clf_pca.fit(x_train_2, y_train)

# Predictions
predictions_test = clf_pca.predict(x_test_2)
predictions_test[predictions_test <= 0.5] = 0
predictions_test[predictions_test > 0.5] = 1
predictions_test = predictions_test.astype(int)

# Scatter plot
y_train_2 = y_train.tolist()
plt.figure(figsize=(8, 6))
plt.scatter(x_train_2[:, 0], x_train_2[:, 1], c=y_train_2, s=10, cmap='viridis')

# Create a mesh to plot in
first_dimension_min, first_dimension_max = x_train_2[:, 0].min() - 1, x_train_2[:, 0].max() + 1
second_dimension_min, second_dimension_max = x_train_2[:, 1].min() - 1, x_train_2[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.arange(first_dimension_min, first_dimension_max, 0.02),
    np.arange(second_dimension_min, second_dimension_max, 0.02)
)

Z = clf_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z[Z <= 0.5] = 0
Z[Z > 0.5] = 1
Z = Z.astype(int)
Z = Z.reshape(xx.shape)

plt.contour(xx, yy, Z)
plt.title('Ridge Regression')
plt.axis('off')
plt.show()


# 3. Graph for Lasso Regression
clf_pca = linear_model.Lasso(alpha=0.1, random_state=0)
clf_pca = clf_pca.fit(x_train_2, y_train)

# Predictions
predictions_test = clf_pca.predict(x_test_2)
predictions_test[predictions_test <= 0.5] = 0
predictions_test[predictions_test > 0.5] = 1
predictions_test = predictions_test.astype(int)

# Scatter plot
y_train_2 = y_train.tolist()
plt.figure(figsize=(8, 6))
plt.scatter(x_train_2[:, 0], x_train_2[:, 1], c=y_train_2, s=10, cmap='viridis')

# Create a mesh to plot in
first_dimension_min, first_dimension_max = x_train_2[:, 0].min() - 1, x_train_2[:, 0].max() + 1
second_dimension_min, second_dimension_max = x_train_2[:, 1].min() - 1, x_train_2[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.arange(first_dimension_min, first_dimension_max, 0.02),
    np.arange(second_dimension_min, second_dimension_max, 0.02)
)

Z = clf_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z[Z <= 0.5] = 0
Z[Z > 0.5] = 1
Z = Z.astype(int)
Z = Z.reshape(xx.shape)

plt.contour(xx, yy, Z)
plt.title('Lasso Regression')
plt.axis('off')
plt.show()


# 4. Graph for Logistic Regression
from sklearn.linear_model import LogisticRegression

clf_pca = LogisticRegression(C=2, max_iter=1000)
clf_pca = clf_pca.fit(x_train_2, y_train)

# Predictions
predictions_test = clf_pca.predict(x_test_2)

# Scatter plot
y_train_2 = y_train.tolist()
plt.figure(figsize=(8, 6))
plt.scatter(x_train_2[:, 0], x_train_2[:, 1], c=y_train_2, s=10, cmap='viridis')

# Create a mesh to plot in
first_dimension_min, first_dimension_max = x_train_2[:, 0].min() - 1, x_train_2[:, 0].max() + 1
second_dimension_min, second_dimension_max = x_train_2[:, 1].min() - 1, x_train_2[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.arange(first_dimension_min, first_dimension_max, 0.02),
    np.arange(second_dimension_min, second_dimension_max, 0.02)
)

Z = clf_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contour(xx, yy, Z)
plt.title('Logistic Regression')
plt.axis('off')
plt.show()


# 5. Graph for SVM classifier with linear kernel
from sklearn.svm import SVC

clf_pca = SVC(C=1.0, kernel='linear')
clf_pca = clf_pca.fit(x_train_2, y_train)

# Predictions
predictions_test = clf_pca.predict(x_test_2)

# Scatter plot
y_train_2 = y_train.tolist()
plt.figure(figsize=(8, 6))
plt.scatter(x_train_2[:, 0], x_train_2[:, 1], c=y_train_2, s=10, cmap='viridis')

# Create a mesh to plot in
first_dimension_min, first_dimension_max = x_train_2[:, 0].min() - 1, x_train_2[:, 0].max() + 1
second_dimension_min, second_dimension_max = x_train_2[:, 1].min() - 1, x_train_2[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.arange(first_dimension_min, first_dimension_max, 0.02),
    np.arange(second_dimension_min, second_dimension_max, 0.02)
)

Z = clf_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contour(xx, yy, Z)
plt.title('SVM Classifier - Linear Kernel')
plt.axis('off')
plt.show()


# 6. Graph for SVM classifier with poly kernel
clf_pca = SVC(C=1.0, kernel='poly', degree=3)
clf_pca = clf_pca.fit(x_train_2, y_train)

# Predictions
predictions_test = clf_pca.predict(x_test_2)

# Scatter plot
y_train_2 = y_train.tolist()
plt.figure(figsize=(8, 6))
plt.scatter(x_train_2[:, 0], x_train_2[:, 1], c=y_train_2, s=10, cmap='viridis')

# Create a mesh to plot in
first_dimension_min, first_dimension_max = x_train_2[:, 0].min() - 1, x_train_2[:, 0].max() + 1
second_dimension_min, second_dimension_max = x_train_2[:, 1].min() - 1, x_train_2[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.arange(first_dimension_min, first_dimension_max, 0.02),
    np.arange(second_dimension_min, second_dimension_max, 0.02)
)

Z = clf_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contour(xx, yy, Z)
plt.title('SVM Classifier - Poly Kernel')
plt.axis('off')
plt.show()


# 7. Graph for SVM classifier with sigmoid kernel
clf_pca = SVC(C=1.0, kernel='sigmoid')
clf_pca = clf_pca.fit(x_train_2, y_train)

# Predictions
predictions_test = clf_pca.predict(x_test_2)

# Scatter plot
y_train_2 = y_train.tolist()
plt.figure(figsize=(8, 6))
plt.scatter(x_train_2[:, 0], x_train_2[:, 1], c=y_train_2, s=10, cmap='viridis')

# Create a mesh to plot in
first_dimension_min, first_dimension_max = x_train_2[:, 0].min() - 1, x_train_2[:, 0].max() + 1
second_dimension_min, second_dimension_max = x_train_2[:, 1].min() - 1, x_train_2[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.arange(first_dimension_min, first_dimension_max, 0.02),
    np.arange(second_dimension_min, second_dimension_max, 0.02)
)

Z = clf_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contour(xx, yy, Z)
plt.title('SVM Classifier - Sigmoid Kernel')
plt.axis('off')
plt.show()


# 8. Graph for SVM classifier with rbf kernel
clf_pca = SVC(C=1.0, kernel='rbf')
clf_pca = clf_pca.fit(x_train_2, y_train)

# Predictions
predictions_test = clf_pca.predict(x_test_2)

# Scatter plot
y_train_2 = y_train.tolist()
plt.figure(figsize=(8, 6))
plt.scatter(x_train_2[:, 0], x_train_2[:, 1], c=y_train_2, s=10, cmap='viridis')

# Create a mesh to plot in
first_dimension_min, first_dimension_max = x_train_2[:, 0].min() - 1, x_train_2[:, 0].max() + 1
second_dimension_min, second_dimension_max = x_train_2[:, 1].min() - 1, x_train_2[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.arange(first_dimension_min, first_dimension_max, 0.02),
    np.arange(second_dimension_min, second_dimension_max, 0.02)
)

Z = clf_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contour(xx, yy, Z)
plt.title('SVM Classifier - RBF Kernel')
plt.axis('off')
plt.show()


# 9. Graph for Naïve Bayes
from sklearn.naive_bayes import GaussianNB

clf_pca = GaussianNB()
clf_pca = clf_pca.fit(x_train_2, y_train)

# Predictions
predictions_test = clf_pca.predict(x_test_2)

# Scatter plot
y_train_2 = y_train.tolist()
plt.figure(figsize=(8, 6))
plt.scatter(x_train_2[:, 0], x_train_2[:, 1], c=y_train_2, s=10, cmap='viridis')

# Create a mesh to plot in
first_dimension_min, first_dimension_max = x_train_2[:, 0].min() - 1, x_train_2[:, 0].max() + 1
second_dimension_min, second_dimension_max = x_train_2[:, 1].min() - 1, x_train_2[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.arange(first_dimension_min, first_dimension_max, 0.02),
    np.arange(second_dimension_min, second_dimension_max, 0.02)
)

Z = clf_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contour(xx, yy, Z)
plt.title('Naïve Bayes')
plt.axis('off')
plt.show()


# 10. Graph for Decision Tree classifier
from sklearn import tree

clf_pca = tree.DecisionTreeClassifier(
    max_depth=None,
    min_samples_split=7,
    min_samples_leaf=2,
    random_state=42
)
clf_pca = clf_pca.fit(x_train_2, y_train)

# Predictions
predictions_test = clf_pca.predict(x_test_2)

# Scatter plot
y_train_2 = y_train.tolist()
plt.figure(figsize=(8, 6))
plt.scatter(x_train_2[:, 0], x_train_2[:, 1], c=y_train_2, s=10, cmap='viridis')

# Create a mesh to plot in
first_dimension_min, first_dimension_max = x_train_2[:, 0].min() - 1, x_train_2[:, 0].max() + 1
second_dimension_min, second_dimension_max = x_train_2[:, 1].min() - 1, x_train_2[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.arange(first_dimension_min, first_dimension_max, 0.02),
    np.arange(second_dimension_min, second_dimension_max, 0.02)
)

Z = clf_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contour(xx, yy, Z)
plt.title('Decision Tree Classifier')
plt.axis('off')
plt.show()


# 11. Graph for Random Forest classifier
from sklearn.ensemble import RandomForestClassifier

clf_pca = RandomForestClassifier(
    max_depth=None,
    min_samples_split=7,
    min_samples_leaf=2,
    n_estimators=100,
    random_state=42
)
clf_pca = clf_pca.fit(x_train_2, y_train)

# Predictions
predictions_test = clf_pca.predict(x_test_2)

# Scatter plot
y_train_2 = y_train.tolist()
plt.figure(figsize=(8, 6))
plt.scatter(x_train_2[:, 0], x_train_2[:, 1], c=y_train_2, s=10, cmap='viridis')

# Create a mesh to plot in
first_dimension_min, first_dimension_max = x_train_2[:, 0].min() - 1, x_train_2[:, 0].max() + 1
second_dimension_min, second_dimension_max = x_train_2[:, 1].min() - 1, x_train_2[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.arange(first_dimension_min, first_dimension_max, 0.02),
    np.arange(second_dimension_min, second_dimension_max, 0.02)
)

Z = clf_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contour(xx, yy, Z)
plt.title('Random Forest Classifier')
plt.axis('off')
plt.show()


# 12. Graph for K Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

clf_pca = KNeighborsClassifier(n_neighbors=5)
clf_pca = clf_pca.fit(x_train_2, y_train)

# Predictions
predictions_test = clf_pca.predict(x_test_2)

# Scatter plot
y_train_2 = y_train.tolist()
plt.figure(figsize=(8, 6))
plt.scatter(x_train_2[:, 0], x_train_2[:, 1], c=y_train_2, s=10, cmap='viridis')

# Create a mesh to plot in
first_dimension_min, first_dimension_max = x_train_2[:, 0].min() - 1, x_train_2[:, 0].max() + 1
second_dimension_min, second_dimension_max = x_train_2[:, 1].min() - 1, x_train_2[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.arange(first_dimension_min, first_dimension_max, 0.02),
    np.arange(second_dimension_min, second_dimension_max, 0.02)
)

Z = clf_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contour(xx, yy, Z)
plt.title('K Nearest Neighbors')
plt.axis('off')
plt.show()