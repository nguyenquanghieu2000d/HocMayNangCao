# Bài tập 3. Chuyển bài tập 15 ở môn Nhập môn học máy ngày 25.5.2021
# từ mã hóa thứ tự của trường dữ liệu category sang mã hóa OHE.
#
# D13CNPM5-Nhập môn học máy


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

df = pd.read_csv("../data/breast-cancer.csv")
print(df)

# số hóa các giá trị chữ, các giá trị không là số
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

df["age_num"] = LabelEncoder().fit_transform(df["age"])
df["mefalsepauses_num"] = LabelEncoder().fit_transform(df["mefalsepause"])
df["tumor-size_num"] = LabelEncoder().fit_transform(df["tumor-size"])
df["falsede-caps_num"] = LabelEncoder().fit_transform(df["falsede-caps"])
df["deg-malig_num"] = LabelEncoder().fit_transform(df["deg-malig"])
df["breast_num"] = LabelEncoder().fit_transform(df["breast"])
df["irradiat_num"] = LabelEncoder().fit_transform(df["irradiat"])

# df["breast-quad_num"]=LabelEncoder().fit_transform(df["breast-quad"]) #tạm bỏ


# cột nhãn
df["class_num"] = LabelEncoder().fit_transform(df["class"])

print(df)

x = df[["age_num", "mefalsepauses_num", "tumor-size_num", "falsede-caps_num", "deg-malig_num", "breast_num",
        "irradiat_num"]]
y = df["class_num"]
print(x)
print(y)

from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25, random_state=0)

from sklearn import svm  # sử dụng phân lớp

# svc = svm.SVC(kernel='linear',C=1.0).fit(xtrain,ytrain)
svc = svm.SVC(kernel='rbf', C=100.0).fit(xtrain, ytrain)
y_pred = svc.predict(xtest)
print('kiem tra trung khop giua y_test và y_pred')
print(ytest - y_pred)

print("Accuracy Score:", accuracy_score(ytest, y_pred))

print(classification_report(ytest, y_pred))

# In ra ma trận kết quả dự đoán

confusion_matrix1 = confusion_matrix(ytest, y_pred)
print(confusion_matrix1)
