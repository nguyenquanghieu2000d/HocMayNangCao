# Bài 1. Xây dựng mô hình học máy với Neural Nework hoặc Decision Tree cho nhận dạng loài hoa Iris sử dụng mã hóa OHE thay cho mã hóa LableEncoding của sklearn


# Iris_biendoi.csv
# Excel
#
# Sử dụng bài mẫu sau:
# "Sử dụng thuật toán thuộc họ SVM, xây dựng mô hình học máy dự báo 3 nhãn hoa tự động của các loài hoa Iris"
#
#
# Iris.csv
# Excel
############
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split  # chú ý
import matplotlib.pyplot as plt

# 10a
df = pd.read_csv('../data/iris.csv')
print(df)
df["Species_num"] = LabelEncoder().fit_transform(df["Species"])
print(df)

x = df[['SepalWidthCm', 'SepalLengthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['Species_num']

x0 = x[y == 0]
x1 = x[y == 1]
x2 = x[y == 2]
plt.plot(x0['SepalLengthCm'], x0['SepalWidthCm'], 'b^', markersize=4, alpha=.8)
plt.plot(x1['SepalLengthCm'], x1['SepalWidthCm'], 'go', markersize=4, alpha=.8)
plt.plot(x2['SepalLengthCm'], x1['SepalWidthCm'], 'r^', markersize=4, alpha=.8)

plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Nhãn lớp 0,lớp 1 và lớp 2')
plt.plot()
plt.show()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print(x_train)
print(y_train)

from sklearn.neural_network import MLPClassifier
nn = MLPClassifier(random_state=1, max_iter=10000).fit(x_train, y_train)
y_out=nn.predict(x_test)
print(nn.score(x_test, y_test))
