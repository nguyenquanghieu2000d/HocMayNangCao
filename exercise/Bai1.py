# Bài tập 2. Sử dụng thư viện pandas đọc vào file Iris.csv được biến bộ nhớ df.
# 2a. Hiển thị df.
# 2b. Chuyển cột nhãn y Species thành dạng dữ liệu mã hóa OHE. Hiển thị dữ liệu được mã hóa.
# 2c. Tạo ra cột vector đầu vào x, và cột nhãn vector đầu ra y của df. Hiển thị x và y.
# G:
import pandas as pd

# 2a
df = pd.read_csv("../data/Iris.csv")
print(df)
# 2b
one_hot_encoded_data = pd.get_dummies(df, columns=['Species'])
print(one_hot_encoded_data)
# 2c
x = df[['SepalWidthCm', 'SepalLengthCm', 'PetalLengthCm', 'PetalWidthCm']]
print(x)
y = one_hot_encoded_data[['Species_Iris-setosa', 'Species_Iris-versicolor', 'Species_Iris-virginica']]
print(y)
