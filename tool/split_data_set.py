# 将数据集按一定比例划分为训练集、验证集和测试集
import os
import random
import shutil
from pathlib import Path, PurePosixPath

# 源文件夹
source_folder = PurePosixPath(Path(__file__).parent.parent, 'resources/flower_photos')
# 目标文件夹
target_folder = PurePosixPath(Path(__file__).parent.parent, 'resources/flower_photos_split')
# 训练集、验证集、测试集比例
train_scale = 0.8
val_scale = 0.15
test_scale = 0.05


def split():
    print('开始数据集划分')
    # 获取类名
    class_names = os.listdir(source_folder)
    # 如果目标文件夹存在，则删除
    if os.path.isdir(target_folder):
        shutil.rmtree(target_folder)
    # 创建目标文件夹
    os.makedirs(target_folder)
    # 在目标目录下创建文件夹
    split_names = ['train', 'val', 'test']
    for split_name in split_names:
        split_path = os.path.join(target_folder, split_name)
        os.mkdir(split_path)
        # 分别在训练集、验证集和测试集创建类别文件夹
        for class_name in class_names:
            class_split_path = os.path.join(split_path, class_name)
            os.mkdir(class_split_path)
    # 按照比例划分数据集，并进行数据图片的复制
    for class_name in class_names:
        # 源文件夹类名路径
        source_path = os.path.join(source_folder, class_name)
        # 源文件夹花朵图片列表
        source_files = os.listdir(source_path)
        # 把图片进行随机打乱
        random.shuffle(source_files)
        # 目标文件夹训练集、验证集、测试集路径
        train_folder = os.path.join(target_folder, 'train', class_name)
        val_folder = os.path.join(target_folder, 'val', class_name)
        test_folder = os.path.join(target_folder, 'test', class_name)

        train_num = int(len(source_files) * train_scale)
        val_num = int(len(source_files) * val_scale)
        test_num = len(source_files) - train_num - val_num
        for i, file in enumerate(source_files):
            if i < train_num:
                shutil.copy2(f'{source_path}/{file}', train_folder)
            if train_num <= i < train_num + val_num:
                shutil.copy2(f'{source_path}/{file}', val_folder)
            elif i >= train_num + val_num:
                shutil.copy2(f'{source_path}/{file}', test_folder)
        print('*' * 25 + class_name + '*' * 25)
        print(f'{class_name}类按照{train_scale}：{val_scale}：{test_scale}的比例划分完成，一共{len(source_files)}张图片')
        print(f'训练集{train_folder}：{train_num}张')
        print(f'验证集{val_folder}：{val_num}张')
        print(f'测试集{test_folder}：{test_num}张')


if __name__ == '__main__':
    split()
