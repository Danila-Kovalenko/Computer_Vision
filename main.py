import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# =====================================================
# 1) Подготовка данных XOR
# =====================================================
X_data = torch.tensor([[0., 0.],
                       [0., 1.],
                       [1., 0.],
                       [1., 1.]], dtype=torch.float32)
y_data = torch.tensor([[0.],
                       [1.],
                       [1.],
                       [0.]], dtype=torch.float32)

# =====================================================
# 2) Определение моделей
# =====================================================

# ----- (A) Простой персептрон (Single Layer Perceptron) -----
class SinglePerceptron(nn.Module):
    def __init__(self):
        super(SinglePerceptron, self).__init__()
        # Один слой, выход 1, сигмоида
        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))


# ----- (B) Многослойный персептрон (MLP) -----
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # Скрытый слой: 4 нейрона, ReLU
        self.hidden = nn.Linear(2, 4)
        # Выходной слой: 1 нейрон, сигмоида
        self.output = nn.Linear(4, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x

# Инициализация двух моделей
model_single = SinglePerceptron()
model_mlp = MLP()

# =====================================================
# 3) Настройка параметров обучения
# =====================================================
# Общие гиперпараметры
learning_rate = 0.01
epochs_single = 100
epochs_mlp = 200

# Оптимизаторы (можно взять SGD, Adam и т.п.)
optimizer_single = optim.SGD(model_single.parameters(), lr=learning_rate)
optimizer_mlp = optim.Adam(model_mlp.parameters(), lr=learning_rate)

# Функция потерь
criterion = nn.BCELoss()

# Для сохранения истории обучения
loss_history_single = []
acc_history_single = []
loss_history_mlp = []
acc_history_mlp = []

# =====================================================
# 4) Обучение простого персептрона
# =====================================================
for epoch in range(epochs_single):
    # Прямой проход
    y_pred = model_single(X_data)
    # Подсчёт ошибки
    loss = criterion(y_pred, y_data)
    
    # Обнуление градиентов
    optimizer_single.zero_grad()
    # Обратный проход (backprop)
    loss.backward()
    # Шаг оптимизации
    optimizer_single.step()
    
    # Сохраняем loss
    loss_history_single.append(loss.item())
    
    # Вычисляем точность
    # Предсказание > 0.5 => класс 1, иначе 0
    predicted = (y_pred >= 0.5).float()
    correct = (predicted == y_data).sum().item()
    accuracy = correct / len(y_data)
    acc_history_single.append(accuracy)

# Итоговая оценка простого персептрона
with torch.no_grad():
    y_pred_final_single = model_single(X_data)
    predicted_single = (y_pred_final_single >= 0.5).float()
    correct_single = (predicted_single == y_data).sum().item()
    acc_single = correct_single / len(y_data)

print("=== Простой персептрон ===")
print(f"Итоговая точность на XOR: {acc_single:.4f}")

# =====================================================
# 5) Обучение многослойного персептрона (MLP)
# =====================================================
for epoch in range(epochs_mlp):
    # Прямой проход
    y_pred = model_mlp(X_data)
    # Подсчёт ошибки
    loss = criterion(y_pred, y_data)
    
    # Обнуление градиентов
    optimizer_mlp.zero_grad()
    # Обратный проход (backprop)
    loss.backward()
    # Шаг оптимизации
    optimizer_mlp.step()
    
    # Сохраняем loss
    loss_history_mlp.append(loss.item())
    
    # Точность
    predicted = (y_pred >= 0.5).float()
    correct = (predicted == y_data).sum().item()
    accuracy = correct / len(y_data)
    acc_history_mlp.append(accuracy)

# Итоговая оценка MLP
with torch.no_grad():
    y_pred_final_mlp = model_mlp(X_data)
    predicted_mlp = (y_pred_final_mlp >= 0.5).float()
    correct_mlp = (predicted_mlp == y_data).sum().item()
    acc_mlp = correct_mlp / len(y_data)

print("\n=== Многослойный персептрон (MLP) ===")
print(f"Итоговая точность на XOR: {acc_mlp:.4f}")

# =====================================================
# 6) Визуализация результатов
# =====================================================
# (A) График потерь и точности для простого персептрона
plt.plot(range(1, epochs_single + 1), loss_history_single, label='Loss (Single Perceptron)')
plt.title('Простой персептрон: Функция потерь по эпохам')
plt.xlabel('Эпоха')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(range(1, epochs_single + 1), acc_history_single, label='Accuracy (Single Perceptron)')
plt.title('Простой персептрон: Точность по эпохам')
plt.xlabel('Эпоха')
plt.ylabel('Точность')
plt.legend()
plt.show()

# (B) График потерь и точности для MLP
plt.plot(range(1, epochs_mlp + 1), loss_history_mlp, label='Loss (MLP)')
plt.title('MLP: Функция потерь по эпохам')
plt.xlabel('Эпоха')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(range(1, epochs_mlp + 1), acc_history_mlp, label='Accuracy (MLP)')
plt.title('MLP: Точность по эпохам')
plt.xlabel('Эпоха')
plt.ylabel('Точность')
plt.legend()
plt.show()
