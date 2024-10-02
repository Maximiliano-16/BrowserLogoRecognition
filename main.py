import numpy as np


class MLP:
    def __init__(self, input_size, hidden_layers, hidden_units, output_size, activation_func, epochs):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.output_size = output_size
        self.epochs = epochs
        self.activation_func = activation_func
        self.weights = []
        self.biases = []
        self.initialize_parameters()

    def initialize_parameters(self):
        layer_sizes = [self.input_size] + [self.hidden_units] * self.hidden_layers + [self.output_size]
        for i in range(len(layer_sizes) - 1):
            weight = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01
            bias = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(weight)
            self.biases.append(bias)

    def activation(self, z):
        if self.activation_func == 'relu':
            return np.maximum(0, z)
        elif self.activation_func == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif self.activation_func == 'tanh':
            return np.tanh(z)
        else:
            raise ValueError("Unsupported activation function.")

    def activation_derivative(self, a):
        if self.activation_func == 'relu':
            return np.where(a > 0, 1, 0)
        elif self.activation_func == 'sigmoid':
            return a * (1 - a)
        elif self.activation_func == 'tanh':
            return 1 - a ** 2
        else:
            raise ValueError("Unsupported activation function.")

    def forward_propagation(self, X):
        self.a = [X]
        for i in range(len(self.weights)):
            z = np.dot(self.a[i], self.weights[i]) + self.biases[i]
            a = self.activation(z)
            self.a.append(a)
        return self.a[-1]

    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        log_likelihood = -np.log(y_pred[range(m), np.argmax(y_true, axis=1)])
        loss = np.sum(log_likelihood) / m
        return loss

    def backpropagation(self, X, y_true):
        m = X.shape[0]
        y_pred = self.a[-1]
        dZ = y_pred - y_true

        dW = [np.dot(self.a[i].T, dZ) / m for i in
              range(len(self.weights) - 1, -1, -1)]
        db = [np.sum(dZ, axis=0, keepdims=True) / m]

        for i in range(len(self.weights) - 1, 0, -1):
            dZ = np.dot(dZ, self.weights[i].T) * self.activation_derivative(
                self.a[i])
            db.insert(0, np.sum(dZ, axis=0, keepdims=True) / m)
            if i > 1:
                dW.insert(0, np.dot(self.a[i - 1].T, dZ) / m)

        return dW, db

    def update_parameters(self, dW, db, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * dW[i]
            self.biases[i] -= learning_rate * db[i]

    def fit(self, X_train, y_train, X_val, y_val, learning_rate=0.01):
        # print(len(self.weights[]))
        for epoch in range(self.epochs):
            y_pred_train = self.forward_propagation(X_train)
            # print(y_pred_train)
            # print('after forwa')
            loss = self.compute_loss(y_train, y_pred_train)
            # print('after loss')
            dW, db = self.backpropagation(X_train, y_train)
            # print('after back')
            self.update_parameters(dW, db, learning_rate)
            # print('after update2')
            # Validation
            y_pred_val = self.forward_propagation(X_val)
            val_loss = self.compute_loss(y_val, y_pred_val)

            print(
                f'Epoch {epoch + 1}/{self.epochs}, Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}')


def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]


if __name__ == "__main__":
    # Гиперпараметры
    input_size = 784  # Например, для изображений 28x28
    hidden_layers = 2
    hidden_units = 16
    output_size = 10  # Например, для цифр от 0 до 9
    epochs = 2
    activation_func = 'relu'

    # model = MLP(input_size, hidden_layers, hidden_units, output_size,
    #             activation_func, epochs)
    # print(model.weights)
    # model.fit(X_train, y_train, X_val, y_val)
    X_train = np.random.rand(1000, input_size)  # Примерные данные
    y_train = one_hot_encode(np.random.randint(0, output_size, 1000),
                             output_size)
    X_val = np.random.rand(200, input_size)  # Примерные данные
    y_val = one_hot_encode(np.random.randint(0, output_size, 200), output_size)

    model = MLP(input_size, hidden_layers, hidden_units, output_size,
                activation_func, epochs)
    model.fit(X_train, y_train, X_val, y_val)

    print('hello')

