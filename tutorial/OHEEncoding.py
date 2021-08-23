import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm  # sử dụng phân lớp
from sklearn.model_selection import train_test_split  # chú ý
from sklearn.preprocessing import LabelEncoder

# 10a
df = pd.read_csv('../data/Iris.csv')
print(df)
# 10b: chuyển đổi giá trị phạm trù sang nhãn số cho thuận lợi tính toán khi cần
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
# 10c.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print(x_train)
print(y_train)
# 10d


svc = svm.SVC(kernel='linear', C=1.0).fit(x_train, y_train)
y_pred = svc.predict(x_test)
print('kiem tra trung khop giua y_test và y_pred')
print(y_test - y_pred)

# score_train=svc.score(x_train, y_train)
# print('SVM:score train:',score_train*100,'%')
# score_test=svc.score(x_test, y_test)
# print('SVM:score test:',score_test*100,'%')
