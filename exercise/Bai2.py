# Bài tập 2. Sử dụng thư viện pandas đọc vào file Iris.csv được biến bộ nhớ df.
# 2a. Hiển thị df.
# 2b. Chuyển cột nhãn y Species thành dạng dữ liệu mã hóa OHE. Hiển thị dữ liệu được mã hóa.
# 2c. Tạo ra cột vector đầu vào x, và cột nhãn vector đầu ra y của df. Hiển thị x và y.
# 2d. Chia tập dữ liệu (x,y) thành tập  (x_train,y_train), (x_test, y_y_test)    ngẫu nhiên tỷ lệ 80%-20%
# Hiển thị tập dữ liệu train và test.
# 2e. Xây dựng mô hình mạng NeuralNetwork học (x_train,y_train). Kiểm tra dự đoán trên tập (x_test,y_test).
#
# G:
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 2a
df = pd.read_csv("../data/Iris.csv")
print(df)
# 2b
print("2b")
one_hot_encoded_data = pd.get_dummies(df, columns=['Species'])
print(one_hot_encoded_data)
# 2c
x = df[['SepalWidthCm', 'SepalLengthCm', 'PetalLengthCm', 'PetalWidthCm']]
print(x)
y = one_hot_encoded_data[['Species_Iris-setosa', 'Species_Iris-versicolor', 'Species_Iris-virginica']]
print(y)
# 2d


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print(x_train)
print(y_train)


# 2e
class NeuralNetwork():
    def __init__(self):
        # seeding for random number generation
        np.random.seed(1)
        # converting weights to a 3 by 1 matrix with values from -1 to 1 and mean of 0
        self.synaptic_weights = 2 * np.random.random((4, 3)) - 1
        # (1,0,0)='setosa', (0,1,0) = 'vesicolor', (0,0,1) = virginica
    def sigmoid(self, x):
        # applying the sigmoid function
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        # computing derivative to the Sigmoid function
        return x * (1 - x)

    def think(self, inputs):
        # pass inputs through our single neuron(our single neuron)
        return self.sigmoid(np.dot(inputs, self.synaptic_weights))

    def fit(self, training_inputs, training_outputs, training_iterations):
        # training the model to make accurate predictions while adjusting weights continually
        for iteration in range(training_iterations):
            # siphon the training data via  the neuron
            output = self.think(training_inputs)
            # computing error rate for back-propagation
            error = training_outputs - output
            # performing weight adjustments
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
            self.synaptic_weights += adjustments

    def predict(self, inputs):
        # passing the inputs via the neuron to get output
        # converting values to floats

        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output


# sử dụng mạng Nơ ron
NN = NeuralNetwork()
print("Beginning Randomly Generated Weights: ")

print(NN.synaptic_weights)
# training data consisting of 4 examples--4 input values and 3 output

# training taking place
NN.fit(x, y, 10000)
# test thử 1 ví dụ cụ thể
print(NN.predict(np.array([4.8, 3.0, 1.4, 0.1])))
