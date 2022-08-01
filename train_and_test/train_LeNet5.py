# cnn模型训练代码，训练的代码会保存在models目录下，折线图会保存在results目录下
# 忽略硬件加速信息
import os
import time
import matplotlib.pyplot as plt
from tool.split_data_set import split
from tool.sqlite import delete, insert

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# 设置训练集目录、验证集目录、图片尺寸、batch_size、epochs
train_folder = '../resources/flower_photos_split/train'
val_folder = '../resources/flower_photos_split/val'
img_size = 224
batch_size = 25
epochs = 20
# 是否需要重新划分数据集，True为重新划分
need_split = False


# 加载训练集和验证集
def data_load():
    if need_split:
        split()
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory=train_folder,
        label_mode='categorical',
        seed=123,
        image_size=(img_size, img_size),
        batch_size=batch_size)
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory=val_folder,
        label_mode='categorical',
        seed=123,
        image_size=(img_size, img_size),
        batch_size=batch_size)
    # 获取类名
    class_names = train_ds.class_names
    if need_split:
        # 把类名写入数据库
        write_db(class_names)
    # 把图像数据存储到数据中，减少训练时间
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return train_ds, val_ds, class_names


# 构建CNN模型
def model_load(class_num):
    model = tf.keras.models.Sequential([
        # 对模型做归一化的处理，将0-255之间的数字统一处理到0到1之间
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_size, img_size, 3)),
        # 卷积层，该卷积层的输出为32个通道，卷积核的大小是3*3，激活函数为relu
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        # 添加池化层，池化的kernel大小是2*2
        tf.keras.layers.MaxPooling2D(2, 2),
        # 卷积层，输出为64个通道，卷积核大小为3*3，激活函数为relu
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        # 池化层，最大池化，对2*2的区域进行池化操作
        tf.keras.layers.MaxPooling2D(2, 2),
        # 将二维的输出转化为一维
        tf.keras.layers.Flatten(),
        # 全连接层
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(class_num, activation='softmax')
    ])
    # 指明模型的训练参数，优化器为sgd优化器，损失函数为交叉熵损失函数
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train(train_ds, val_ds, model):
    # 开始训练，记录开始时间
    begin_time = time.time()
    # 指明训练的轮数epoch，开始训练
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=1)
    # 保存模型
    model.save('../resources/models/LeNet5.h5')
    # 输出训练用时
    print(f'该循环程序运行时间：{round(time.time() - begin_time, 2)}秒')
    model.summary()
    return history


# 展示训练过程的曲线
def show_loss_acc(history):
    # 从history中提取模型训练集和验证集准确率信息和误差信息
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # 按照上下结构将图画输出
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.title('Training/Validation Accuracy')
    plt.xlim(0, epochs)
    plt.ylim(min(plt.ylim()), 1)

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.title('Training/Validation Loss')
    plt.xlim(0, epochs)
    plt.savefig('../resources/results/LeNet5_results.png', dpi=100)


# 把类名写入数据库
def write_db(class_names):
    data = []
    for i, name in enumerate(class_names):
        data.append((i, name))
    # 先把原来的数据清空再重新写入
    delete('delete from labels')
    sql = 'insert into labels (id, name) values (?, ?)'
    insert(sql, data)


def main():
    train_ds, val_ds, class_names = data_load()
    model = model_load(len(class_names))
    history = train(train_ds, val_ds, model)
    show_loss_acc(history)


if __name__ == '__main__':
    main()
