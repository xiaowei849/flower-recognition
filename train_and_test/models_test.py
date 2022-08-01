# 模型测试代码，测试会生成热力图，热力图会保存在resources/results目录下
import os
import matplotlib.pyplot as plt
import numpy as np

# 忽略硬件加速信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# 数据集目录、图片尺寸、batch_size
train_folder = '../resources/flower_photos_split/train'
val_folder = '../resources/flower_photos_split/val'

img_size = 224
batch_size = 32


# 加载训练集和验证集
def data_load():
    global train_ds, val_ds, class_names
    # 加载训练集
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_folder,
        label_mode='categorical',
        seed=123,
        image_size=(img_size, img_size),
        batch_size=batch_size)
    # 加载验证集
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        val_folder,
        label_mode='categorical',
        seed=123,
        image_size=(img_size, img_size),
        batch_size=batch_size)
    class_names = train_ds.class_names
    # 返回训练集、验证集和类名
    return train_ds, val_ds, class_names


# 测试模型
def test():
    model_list = ['DenseNet201', 'InceptionV3', 'MobileNetV2', 'DenseNet169', 'MobileNet', 'ResNet152V2', 'Xception',
                  'ResNet101V2', 'InceptionResNetV2', 'NASNetMobile', 'VGG19', 'LeNet5']
    model_list = ['MobileNetV2']
    for i, m in enumerate(model_list):
        # 加载模型
        model = tf.keras.models.load_model(f'../resources/models/{m}.h5')
        # 获取loss和accuracy
        loss, accuracy = model.evaluate(val_ds)
        # 输出测试结果
        print(f'{i + 1}/{len(model_list)} {m} test accuracy: {accuracy}, loss: {loss}')
        # 花朵种类
        class_num = len(class_names)
        # 真实标签和预测标签
        real_labels = []
        predict_labels = []
        for batch_images, batch_labels in val_ds:
            batch_labels = batch_labels.numpy()
            batch_predict = model.predict(batch_images)
            real_labels += np.argmax(batch_labels, axis=1).tolist()
            predict_labels += np.argmax(batch_predict, axis=1).tolist()
        # 热力图
        heat_maps = np.zeros((class_num, class_num))
        for test_real_label, test_predict_label in zip(real_labels, predict_labels):
            heat_maps[test_real_label][test_predict_label] = heat_maps[test_real_label][test_predict_label] + 1
        # 对热力图进行归一化处理
        heat_maps_sum = np.sum(heat_maps, axis=1).reshape(-1, 1)
        heat_maps_float = heat_maps / heat_maps_sum
        # 输出热力图
        show_heatmaps(title=f'{m} heatmap', x_labels=class_names, y_labels=class_names, heat_maps_float=heat_maps_float,
                      save_name=f'../resources/results/{m}_heatmap.png')


def show_heatmaps(title, x_labels, y_labels, heat_maps_float, save_name):
    # 设置matplotlib字体
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 创建一个包含子图区域的画布
    fig, ax = plt.subplots()
    # 设置热力图颜色：深红色到浅红色
    im = ax.imshow(heat_maps_float, cmap='OrRd')
    # 设置刻度和刻度标签
    ax.set_xticks(np.arange(len(y_labels)))
    ax.set_yticks(np.arange(len(x_labels)))
    ax.set_xticklabels(y_labels)
    ax.set_yticklabels(x_labels)
    # 旋转x轴刻度标签标签
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    # 添加每个热力块的具体数值
    for i in range(len(x_labels)):
        for j in range(len(y_labels)):
            ax.text(j, i, round(heat_maps_float[i, j], 2), ha='center', va='center', color='black')
    # 设置x轴标签、y轴标签、标题
    ax.set_xlabel('Predict')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    fig.tight_layout()
    plt.colorbar(im)
    plt.savefig(save_name, dpi=100)
    plt.show()


if __name__ == '__main__':
    train_ds, val_ds, class_names = data_load()
    test()
