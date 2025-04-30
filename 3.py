import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms

# ========================
# 1. ЗАГРУЗКА ДАННЫХ MNIST ЧЕРЕЗ PyTorch
# ========================
transform = transforms.ToTensor()
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Преобразуем в numpy
x_train_full = train_dataset.data.numpy()  # (60000, 28, 28)
y_train_full = train_dataset.targets.numpy()  # (60000,)

x_test = test_dataset.data.numpy()  # (10000, 28, 28)
y_test = test_dataset.targets.numpy()  # (10000,)

# Преобразуем к float32 и нормализуем в [0, 1]
x_train_full = x_train_full.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

# Разворачиваем картинки 28x28 в вектор из 784 значений
x_train_full = x_train_full.reshape(len(x_train_full), 784)
x_test = x_test.reshape(len(x_test), 784)

# Сделаем train/val раздел (примем, например, 20k на обучение, 5k на валидацию)
train_size = 20000
val_size = 5000
x_train = x_train_full[:train_size]
y_train = y_train_full[:train_size]
x_val   = x_train_full[train_size:train_size+val_size]
y_val   = y_train_full[train_size:train_size+val_size]

# Остальное останется для тестирования (x_test, y_test)

# Функция для one-hot кодировки
def one_hot_encode(labels, num_classes=10):
    one_hot = np.zeros((len(labels), num_classes), dtype=np.float32)
    for i, lbl in enumerate(labels):
        one_hot[i, lbl] = 1.0
    return one_hot

y_train_oh = one_hot_encode(y_train, 10)
y_val_oh   = one_hot_encode(y_val, 10)
y_test_oh  = one_hot_encode(y_test, 10)

# ========================
# 2. РЕАЛИЗАЦИЯ НЕЙРОННОЙ СЕТИ (С ОДНИМ СКРЫТЫМ СЛОЕМ) НА NUMPY
# ========================

input_dim = 784      # размер входного вектора
hidden_dim = 128     # нейронов в скрытом слое (регулируйте)
output_dim = 10      # 10 классов
learning_rate = 0.01
epochs = 5
batch_size = 100

# Инициализация весов
W1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * 0.01
b1 = np.zeros((hidden_dim,), dtype=np.float32)

W2 = np.random.randn(hidden_dim, output_dim).astype(np.float32) * 0.01
b2 = np.zeros((output_dim,), dtype=np.float32)

# Активации
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(np.float32)

def softmax(z):
    # z shape: (batch_size, output_dim)
    exps = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

# Функция потерь (кросс-энтропия)
def cross_entropy_loss(y_pred, y_true):
    eps = 1e-9
    return -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1))

# Прямой проход
def forward_pass(x_batch):
    """
    x_batch: shape (batch_size, input_dim)
    Возвращает словарь с промежуточными результатами
    """
    z1 = x_batch.dot(W1) + b1   # (batch_size, hidden_dim)
    a1 = relu(z1)               # ReLU
    z2 = a1.dot(W2) + b2        # (batch_size, output_dim)
    a2 = softmax(z2)            # Softmax
    return {
        'x': x_batch,
        'z1': z1,
        'a1': a1,
        'z2': z2,
        'a2': a2
    }

# Обратный проход
def backward_pass(cache, y_true):
    """
    Возвращает dW1, db1, dW2, db2
    """
    x = cache['x']
    z1 = cache['z1']
    a1 = cache['a1']
    z2 = cache['z2']
    a2 = cache['a2']

    batch_size_local = x.shape[0]

    # dL/dz2
    dz2 = (a2 - y_true)  # (batch_size, output_dim)
    dW2 = a1.T.dot(dz2) / batch_size_local
    db2 = np.sum(dz2, axis=0) / batch_size_local

    da1 = dz2.dot(W2.T)  # (batch_size, hidden_dim)
    dz1 = da1 * relu_derivative(z1)  # (batch_size, hidden_dim)
    dW1 = x.T.dot(dz1) / batch_size_local
    db1 = np.sum(dz1, axis=0) / batch_size_local

    return dW1, db1, dW2, db2

# Обновление параметров
def update_params(dW1, db1, dW2, db2, lr):
    global W1, b1, W2, b2
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

# Функция для вычисления предсказания класса
def predict(x_data):
    forward_out = forward_pass(x_data)
    probs = forward_out['a2']
    return np.argmax(probs, axis=1)

# Точность
def accuracy_score(y_pred, y_true):
    return np.mean(y_pred == y_true)

# Precision и Recall (микро)
def precision_recall_score_micro(y_pred, y_true, num_classes=10):
    TP = 0
    FP = 0
    FN = 0
    for c in range(num_classes):
        pred_c = (y_pred == c)
        true_c = (y_true == c)
        TP_c = np.sum(pred_c & true_c)
        FP_c = np.sum(pred_c & (~true_c))
        FN_c = np.sum((~pred_c) & true_c)
        TP += TP_c
        FP += FP_c
        FN += FN_c
    precision = TP / (TP + FP + 1e-9)
    recall = TP / (TP + FN + 1e-9)
    return precision, recall

# (Опционально) пример функции дропаута
def apply_dropout(a, dropout_rate=0.5):
    mask = (np.random.rand(*a.shape) > dropout_rate).astype(np.float32)
    a_dropped = a * mask / (1.0 - dropout_rate)
    return a_dropped, mask

# ========================
# 3. ОБУЧЕНИЕ
# ========================
num_batches = len(x_train) // batch_size

train_losses = []
val_losses = []

for epoch in range(epochs):
    # Перемешиваем
    indices = np.random.permutation(len(x_train))
    x_train_shuffled = x_train[indices]
    y_train_oh_shuffled = y_train_oh[indices]

    epoch_loss = 0.0

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size

        x_batch = x_train_shuffled[start_idx:end_idx]
        y_batch = y_train_oh_shuffled[start_idx:end_idx]

        # Forward
        cache = forward_pass(x_batch)
        loss = cross_entropy_loss(cache['a2'], y_batch)
        epoch_loss += loss

        # Backward
        dW1, db1, dW2, db2 = backward_pass(cache, y_batch)

        # Update
        update_params(dW1, db1, dW2, db2, learning_rate)

    # Средний лосс за эпоху
    epoch_loss /= num_batches

    # Лосс на валидации
    val_cache = forward_pass(x_val)
    val_loss = cross_entropy_loss(val_cache['a2'], y_val_oh)

    train_losses.append(epoch_loss)
    val_losses.append(val_loss)

    print(f"Эпоха {epoch+1}/{epochs}, Loss (train) = {epoch_loss:.4f}, Loss (val) = {val_loss:.4f}")

# График
plt.plot(range(1, epochs+1), train_losses, label='Training Loss')
plt.plot(range(1, epochs+1), val_losses, label='Validation Loss')
plt.xlabel('Эпоха')
plt.ylabel('Loss')
plt.legend()
plt.title('График потерь на обучающей и валидационной выборках')
plt.show()

# ========================
# 4. ОЦЕНКА НА ТЕСТОВОМ НАБОРЕ
# ========================
y_test_pred = predict(x_test)
acc_test = accuracy_score(y_test_pred, y_test)
prec_micro, rec_micro = precision_recall_score_micro(y_test_pred, y_test)

print("=== РЕЗУЛЬТАТЫ НА ТЕСТОВОЙ ВЫБОРКЕ ===")
print(f"Accuracy = {acc_test:.4f}")
print(f"Precision (micro) = {prec_micro:.4f}")
print(f"Recall (micro) = {rec_micro:.4f}")
