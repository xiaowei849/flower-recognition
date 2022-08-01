# 模型训练代码，训练的代码会保存在models目录下，折线图会保存在results目录下
import os
import time
import matplotlib.pyplot as plt
from tool.split_data_set import split
from tool.sqlite import delete, insert

# 忽略硬件加速信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# 设置训练集目录、验证集目录、图片尺寸、batch_size、epochs
train_folder = '../resources/flower_photos_split/train'
val_folder = '../resources/flower_photos_split/val'
img_size = 224
# 单次传递参数个数
batch_size = 25
# 训练迭代次数
epochs = 15
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


# 模型加载，指定图片处理的大小和是否进行迁移学习
def model_load(class_num):
    # 基础模型
    base_model = tf.keras.applications.DenseNet169(input_shape=(img_size, img_size, 3), include_top=False, weights='imagenet')
    # 将模型的主干参数进行冻结
    base_model.trainable = False
    model = tf.keras.models.Sequential([
        # 将图片像素值从[0，255]重新缩放到[-1，1]
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 127.5, offset=-1, input_shape=(img_size, img_size, 3)),
        # 设置主干模型
        base_model,
        # 对主干模型的输出进行全局平均池化
        tf.keras.layers.GlobalAveragePooling2D(),
        # 通过全连接层映射到最后的分类数目上
        tf.keras.layers.Dense(class_num, activation='softmax')
    ])
    # 模型训练的优化器为adam优化器，模型的损失函数为交叉熵损失函数
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train(train_ds, val_ds, model):
    # 开始训练，记录开始时间
    begin_time = time.time()
    # 指明训练的轮数epoch，开始训练
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    # 保存模型
    model.save('../resources/models/DenseNet169.h5')
    # 输出训练用时
    print(f'该循环程序运行时间：{round(time.time() - begin_time, 2)}秒')
    # 输出模型
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
    plt.savefig('../resources/results/DenseNet169_results.png', dpi=100)


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
