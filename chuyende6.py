import pandas as pd
import numpy as np

from math import sqrt
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('car.data')

for i in range(df.shape[1]):
    unique_values = np.unique(df.iloc[:, i])
    value_map = {unique_values[j]: j for j in range(len(unique_values))}
    df.iloc[:, i] = df.iloc[:, i].map(value_map)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


class KNN:
    def __init__(self, k):
        self.k = k
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        y_pred = []
        for i in range(len(X_test)):
            distances = []
            for j in range(len(self.X_train)):
                distance = np.sqrt(np.sum(np.square(X_test.iloc[i, :] - self.X_train.iloc[j, :])))
                distances.append((distance, self.y_train.iloc[j]))
            distances.sort()
            k_nearest_neighbors = distances[:self.k]
            counts = {}
            for neighbor in k_nearest_neighbors:
                if neighbor[1] in counts:
                    counts[neighbor[1]] += 1
                else:
                    counts[neighbor[1]] = 1
            sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            y_pred.append(sorted_counts[0][0])
        return y_pred

knn_student = KNN(k=5)
knn_student.fit(X_train, y_train)
y_pred_knn_student = knn_student.predict(X_test)
accuracy_knn_student = accuracy_score(y_test, y_pred_knn_student)
print('KNN (sinh viên):', accuracy_knn_student)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn_sklearn = knn.predict(X_test)
accuracy_knn_sklearn = accuracy_score(y_test, y_pred_knn_sklearn)
print('KNN (sklearn):', accuracy_knn_sklearn)

# class DecisionTree:
#     def fit(self, X_train, y_train):
#         self.tree = self.build_tree(X_train, y_train)
    
#     def build_tree(self, X_train, y_train):
#         num_samples, num_features = X_train.shape
        
#         # Tất cả các nhãn trong tập huấn luyện đều thuộc về cùng một lớp
#         if len(np.unique(y_train)) == 1:
#             return y_train[value]
        
#         # Không còn thuộc tính để phân chia
#         if num_features == 0:
#             return np.bincount(y_train).argmax()
        
#         # Chọn thuộc tính tốt nhất để phân chia tập dữ liệu
#         best_feature = self.find_best_split(X_train, y_train)
#         best_feature_name = X_train.columns[best_feature]
#         tree = {best_feature_name: {}}
        
#         # Tạo các nhánh của cây
#         feature_values = np.unique(X_train.iloc[:, best_feature])
#         for value in feature_values:
#             X_subset, y_subset = self.split_data(X_train, y_train, best_feature, value)
#             tree[best_feature_name][value] = self.build_tree(X_subset, y_subset)
        
#         return tree
    
#     def find_best_split(self, X_train, y_train):
#         num_features = X_train.shape[1]
#         best_gain = -1
#         best_feature = -1
        
#         # Tính độ lợi của mỗi thuộc tính
#         for feature in range(num_features):
#             gain = self.calculate_gain(X_train, y_train, feature)
#             if gain > best_gain:
#                 best_gain = gain
#                 best_feature = feature
        
#         return best_feature
    
#     def calculate_gain(self, X_train, y_train, feature):
#         # Tính entropy của tập huấn luyện
#         entropy = self.calculate_entropy(y_train)
        
#         # Tính entropy của các tập con
#         feature_values = np.unique(X_train.iloc[:, feature])
#         weighted_entropy = 0
#         for value in feature_values:
#             X_subset, y_subset = self.split_data(X_train, y_train, feature, value)
#             subset_weight = len(X_subset) / len(X_train)
#             subset_entropy = self.calculate_entropy(y_subset)
#             weighted_entropy += subset_weight * subset_entropy
        
#         # Tính độ lợi
#         gain = entropy - weighted_entropy
#         return gain
    
#     def calculate_entropy(self, y_train):
#         num_samples = len(y_train)
#         _, counts = np.unique(y_train, return_counts=True)
#         probabilities = counts / num_samples
#         entropy = -np.sum(probabilities * np.log2(probabilities))
#         return entropy
    
#     def split_data(self, X_train, y_train, feature, value):
#         mask = X_train.iloc[:, feature] == value
#         X_subset = X_train[mask].drop(X_train.columns[feature], axis=1)
#         y_subset = y_train[mask]
#         return X_subset, y_subset
    
#     def predict(self, X_test):
#         y_pred = []
#         for _, row in X_test.iterrows():
#             y_pred.append(self.traverse_tree(self.tree, row))
#         return y_pred
    
#     def traverse_tree(self, tree, row):
#         for feature, subtree in tree.items():
#             value = row[feature]
#             if value in subtree:
#                 subtree = subtree[value]
#                 if type(subtree) == dict:
#                     return self.traverse_tree(subtree, row)
#                 else:
#                     return subtree


# dt_student = DecisionTree()
# dt_student.fit(X_train, y_train)
# y_pred_dt_student = dt_student.predict(X_test)
# accuracy_dt_student = accuracy_score(y_test, y_pred_dt_student)
# print('Decision Trees (sinh viên):', accuracy_dt_student)