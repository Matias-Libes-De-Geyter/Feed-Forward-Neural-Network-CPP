import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('training_data.csv')
epochs = data['Epoch']
train_acc = data['TrainAcc']
test_acc = data['ValAcc']
loss = data['Loss']

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_acc, marker='o', color='orange', label='Train Accuracy')
plt.plot(epochs, test_acc, marker='o', color='red', label='Test Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid()
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, marker='o', color='blue', label='Training Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()