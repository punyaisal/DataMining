# Import library yang diperlukan
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Load dataset IRIS
iris = datasets.load_iris()
X = iris.data  # Fitur
y = iris.target  # Label

# Membagi dataset menjadi data training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Membuat model Decision Tree dan melatihnya
clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

# Melakukan prediksi pada data testing
y_pred = clf.predict(X_test)

# Menghitung akurasi model
print("Akurasi:", metrics.accuracy_score(y_test, y_pred))

# Visualisasi Decision Tree
plt.figure(figsize=(12,8))
plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.show()
