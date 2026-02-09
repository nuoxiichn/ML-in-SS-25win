import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from keras.layers import Dropout
from keras.callbacks import EarlyStopping


# ---------------------------------------------------------
# 0. Some useful functions
# ---------------------------------------------------------
import matplotlib.pyplot as plt
import os

def plot_history(history, title="", save_dir="../Results_PNG"):
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 生成文件名前缀（移除空格和特殊字符）
    prefix = title.replace(" ", "_").replace("+", "").replace("__", "_").strip("_")
    if not prefix:
        prefix = "history"
    
    # 绘制准确率图
    plt.figure()
    plt.plot(history.history["accuracy"], label="train acc")
    plt.plot(history.history["val_accuracy"], label="val acc")
    plt.title("Accuracy " + title)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    accuracy_path = os.path.join(save_dir, f"{prefix}_accuracy.png")
    plt.savefig(accuracy_path, dpi=300, bbox_inches='tight')
    print(f"Saved accuracy plot to: {accuracy_path}")
    plt.show()

    # 绘制损失图
    plt.figure()
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.title("Loss " + title)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    loss_path = os.path.join(save_dir, f"{prefix}_loss.png")
    plt.savefig(loss_path, dpi=300, bbox_inches='tight')
    print(f"Saved loss plot to: {loss_path}")
    plt.show()

# ---------------------------------------------------------
# 1. Load and Preprocess Data
# ---------------------------------------------------------
print("Loading dataset...")
website_phishing = fetch_ucirepo(id=379)
X_train = website_phishing.data.features.values.astype(np.float32)
y_train = (website_phishing.data.targets.values.flatten() + 1).astype(np.int64) # Map -1,0,1 to 0,1,2
print (X_train,'\n',y_train)
print(X_train.shape)
print(y_train.shape)

num_pixels = X_train.shape[1]
num_classes = len(np.unique(y_train))  # 获取类别数：应该是3 (0, 1, 2)
print(f"Number of classes: {num_classes}")

# ---------------------------------------------------------
# 2. define classification model
# ---------------------------------------------------------

X_train2, X_val, y_train2, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42
)

# 将标签转换为one-hot编码格式
y_train2 = to_categorical(y_train2, num_classes)
y_val = to_categorical(y_val, num_classes)


def deep_model_dropout():
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(num_pixels,)))
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# build the model
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model = deep_model_dropout()
history = model.fit(
    X_train2, y_train2,
    validation_data=(X_val, y_val),
    epochs=80,
    callbacks=[early_stop],
    verbose=2
)

plot_history(history, "Deep + Dropout + EarlyStopping")
