import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

from PIL import Image
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# 超参数设置
IMG_HEIGHT, IMG_WIDTH = 128, 128  # 图像尺寸
BATCH_SIZE = 16  # 减少 batch size 以增加模型更新频率
EPOCHS = 400  # 增加训练轮次，以便更好地拟合
MAX_CELL_COUNT = 2000  # 根据细胞数量分布确定的最大值，用于归一化

# 数据路径和标签文件
data_path = r"C:\Users\l1361\Desktop\result"  # 灰度图片文件夹路径
label_file = r"C:\Users\l1361\Desktop\Summary11.csv" # CSV文件路径，包含图像文件名和标签信息

# 加载标签数据并移除测试图片 test_image.jpg
labels_df = pd.read_csv(label_file)
labels_df = labels_df[labels_df['filename'] != 'test_image.jpg']  # 移除测试图片

# 自定义数据增强函数
def augment_image(image):
    datagen = ImageDataGenerator(
        rotation_range=20,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True,
        zoom_range=0.2,
        fill_mode='nearest'
    )
    augmented_image = datagen.random_transform(image)
    return augmented_image

# 加载图像并应用增强
def load_and_augment_images(data_path, labels_df):
    images = []
    labels = []

    for _, row in labels_df.iterrows():
        img_path = os.path.join(data_path, row['filename'])

        try:
            image = Image.open(img_path).convert("RGB")  # 转换为RGB格式
            image = image.resize((IMG_WIDTH, IMG_HEIGHT))
            image = np.array(image) / 255.0

            # 原始图片
            images.append(image)
            labels.append(row['cell_count'] / MAX_CELL_COUNT)

            # 应用数据增强
            augmented_image = augment_image(image)
            images.append(augmented_image)
            labels.append(row['cell_count'] / MAX_CELL_COUNT)
        except Exception as e:
            print(f"无法读取图片 {img_path}，跳过... 原因: {e}")

    return np.array(images), np.array(labels)

# 加载和增强数据
images, labels = load_and_augment_images(data_path, labels_df)

# 构建TensorFlow数据集对象
train_dataset = tf.data.Dataset.from_tensor_slices((images, labels))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# 构建卷积神经网络模型
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation=layers.PReLU(), input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), kernel_regularizer=regularizers.l2(0.001)),
    layers.Conv2D(32, (3, 3), activation=layers.PReLU(), kernel_regularizer=regularizers.l2(0.001)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation=layers.PReLU(), kernel_regularizer=regularizers.l2(0.001)),
    layers.Conv2D(64, (3, 3), activation=layers.PReLU(), kernel_regularizer=regularizers.l2(0.001)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation=layers.PReLU(), kernel_regularizer=regularizers.l2(0.001)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),  # 添加Dropout层以减少过度拟合
    layers.Flatten(),
    layers.Dense(128, activation=layers.PReLU(), kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.3),  # 再添加Dropout层
    layers.Dense(1, activation='linear')  # 使用线性激活函数以适应回归输出
])

# 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),  # 降低学习率
              loss='mean_squared_error',  # 使用Huber损失函数
              metrics=['mae'])

# 设置Early Stopping回调
early_stopping = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)

# 训练模型
history = model.fit(train_dataset, epochs=EPOCHS, callbacks=[early_stopping])

# 可视化训练过程
plt.plot(history.history['loss'], label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

# 预测 test_image.jpg 中的细胞数量
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict_cell_count(model, image_path):
    processed_image = preprocess_image(image_path)
    predicted_count = model.predict(processed_image)[0][0] * MAX_CELL_COUNT  # 还原归一化的细胞数量
    return max(0, int(predicted_count))  # 确保预测数量不为负

# 替换为 test_image.jpg 的路径
test_image_path = os.path.join(data_path, 'test_image.jpg')
predicted_count = predict_cell_count(model, test_image_path)
print(f"test_image.jpg 中预测的细胞数量: {predicted_count}")