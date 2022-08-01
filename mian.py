# 图形化界面
import os
import shutil
import sys
import time
import cv2
import numpy as np
import requests
import base64
import re
from PIL import Image
from datetime import datetime
from tool.sqlite import check, delete, insert
from tool.excel import save_excel
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon, QPixmap, QStandardItemModel, QStandardItem, QImage
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox, \
    QApplication, QTextBrowser, QTextEdit, QTableView, QAbstractItemView, QComboBox

# 忽略硬件加速信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(1600, 900)
        self.setWindowIcon(QIcon('resources/img/icon.jpg'))
        self.setWindowTitle('花朵图像处理与识别系统')
        self.model = tf.keras.models.load_model('resources/models/DenseNet201.h5')
        self.my_model = 'DenseNet201'
        self.record_num = 0
        self.text_edit = QTextEdit()
        self.table = QStandardItemModel(20, 3)
        self.flower_info = QTextBrowser()
        self.btn_open_camera = QPushButton('打开摄像头')
        self.timer_camera = QTimer()
        self.cap = cv2.VideoCapture()
        self.show_image = None
        self.is_handle = False
        self.mode = None
        self.main_img = None
        self.predict = 'resources/temp/target1.png'
        self.predict_gray = 'resources/temp/target2.png'
        self.predict_cut = 'resources/temp/target3.png'
        self.predict_cut_prospects = 'resources/temp/target4.png'
        # 图片展示
        self.img_example = QLabel()
        self.img_example.setPixmap(QPixmap('resources/img/wait_loading.png').scaled(400, 400, Qt.KeepAspectRatio))
        self.img_example.setAlignment(Qt.AlignCenter)
        self.img1 = QLabel()
        self.img1.setPixmap(QPixmap('resources/img/wait_loading.png').scaled(280, 280, Qt.KeepAspectRatio))
        self.img1.setAlignment(Qt.AlignCenter)
        self.img1_result = QLabel('识别结果：')
        self.img1_accuracy = QLabel('准确率：')
        self.img2 = QLabel()
        self.img2.setPixmap(QPixmap('resources/img/wait_loading.png').scaled(280, 280, Qt.KeepAspectRatio))
        self.img2_result = QLabel('识别结果：')
        self.img2_accuracy = QLabel('准确率：')
        self.img2.setAlignment(Qt.AlignCenter)
        self.img3 = QLabel()
        self.img3.setPixmap(QPixmap('resources/img/wait_handle.png').scaled(280, 280, Qt.KeepAspectRatio))
        self.img3_result = QLabel('识别结果：')
        self.img3_accuracy = QLabel('准确率：')
        self.img3.setAlignment(Qt.AlignCenter)
        self.img4 = QLabel()
        self.img4.setPixmap(QPixmap('resources/img/wait_handle.png').scaled(280, 280, Qt.KeepAspectRatio))
        self.img4_result = QLabel('识别结果：')
        self.img4_accuracy = QLabel('准确率：')
        self.img4.setAlignment(Qt.AlignCenter)
        # 窗口布局
        self.win_layout()
        # 从数据库中删除旧的识别记录数据数据
        delete('delete from records')

    def win_layout(self):
        self.setObjectName('window')
        # 主布局（水平布局）
        main_layout = QHBoxLayout()
        # 主-左（垂直布局）
        main_left_widget = QWidget()
        main_left_widget.setObjectName('border')
        main_left_layout = QVBoxLayout()
        # 传入图片方式
        title1 = QLabel('本地选取图片')
        title1.setAlignment(Qt.AlignCenter)
        title2 = QLabel('图片链接/base64')
        title2.setAlignment(Qt.AlignCenter)
        title3 = QLabel('摄像头抓拍')
        title3.setAlignment(Qt.AlignCenter)
        # 左侧按钮
        btn_load_local = QPushButton('选取图片')
        btn_load_local.setObjectName('btn')
        btn_load_local.clicked.connect(self.load_img)
        btn_load_clear = QPushButton('清除文本')
        btn_load_clear.setObjectName('btn')
        btn_load_clear.clicked.connect(self.text_edit.clear)
        btn_load_url = QPushButton('加载图片')
        btn_load_url.setObjectName('btn')
        btn_load_url.clicked.connect(self.download_img)
        self.btn_open_camera.setObjectName('btn')
        self.btn_open_camera.clicked.connect(self.open_camera)
        btn_load_capture = QPushButton('抓拍图片')
        btn_load_capture.setObjectName('btn')
        btn_load_capture.clicked.connect(self.cap_picture)
        self.timer_camera.timeout.connect(self.show_camera)
        # 预览
        img_msg = QLabel('预览')
        img_msg.setAlignment(Qt.AlignCenter)
        # 网络链接两个按钮水平布局
        btn_url_widget = QWidget()
        btn_url_layout = QHBoxLayout()
        btn_url_layout.addWidget(btn_load_clear)
        btn_url_layout.addWidget(btn_load_url)
        btn_url_widget.setLayout(btn_url_layout)
        # 摄像头两个按钮水平布局
        btn_camera_widget = QWidget()
        btn_camera_layout = QHBoxLayout()
        btn_camera_layout.addWidget(self.btn_open_camera)
        btn_camera_layout.addWidget(btn_load_capture)
        btn_camera_widget.setLayout(btn_camera_layout)
        # 把所有内容放置到左侧窗口
        main_left_layout.addWidget(title1)
        main_left_layout.addWidget(btn_load_local)
        main_left_layout.addWidget(title2)
        main_left_layout.addWidget(self.text_edit)
        main_left_layout.addWidget(btn_url_widget)
        main_left_layout.addWidget(title3)
        main_left_layout.addWidget(btn_camera_widget)
        main_left_layout.addWidget(img_msg)
        main_left_layout.addWidget(self.img_example)
        main_left_widget.setLayout(main_left_layout)

        # 主-中（垂直布局）
        main_mid_widget = QWidget()
        main_mid_widget.setObjectName('border')
        main_mid_layout = QVBoxLayout()
        # 标题
        mid_title = QLabel('不同图像处理花朵识别准确率对比')
        mid_title.setAlignment(Qt.AlignCenter)
        img_title1 = QLabel('原图')
        img_title1.setAlignment(Qt.AlignCenter)
        img_title2 = QLabel('灰度图')
        img_title2.setAlignment(Qt.AlignCenter)
        img_title3 = QLabel('裁剪图')
        img_title3.setAlignment(Qt.AlignCenter)
        img_title4 = QLabel('前景图')
        img_title4.setAlignment(Qt.AlignCenter)
        # 识别结果和准确率
        self.img1_result.setAlignment(Qt.AlignCenter)
        self.img1_accuracy.setAlignment(Qt.AlignCenter)
        self.img2_result.setAlignment(Qt.AlignCenter)
        self.img2_accuracy.setAlignment(Qt.AlignCenter)
        self.img3_result.setAlignment(Qt.AlignCenter)
        self.img3_accuracy.setAlignment(Qt.AlignCenter)
        self.img4_result.setAlignment(Qt.AlignCenter)
        self.img4_accuracy.setAlignment(Qt.AlignCenter)
        # 中间按钮
        btn_handle = QPushButton('图像处理')
        btn_handle.setObjectName('btn')
        btn_handle.clicked.connect(self.handle)
        btn_predict = QPushButton('开始识别')
        btn_predict.setObjectName('btn')
        btn_predict.clicked.connect(self.predict_img)
        # 模型选择下拉列表
        use_model = QComboBox()
        use_model.setObjectName('choose_model')
        use_model.addItems(['DenseNet201', 'InceptionV3', 'MobileNetV2', 'DenseNet169', 'MobileNet', 'ResNet152V2', 'Xception',
                            'ResNet101V2', 'InceptionResNetV2', 'NASNetMobile', 'VGG19', 'LeNet5'])
        use_model.setMaximumWidth(150)
        # 模型选择
        use_model.currentIndexChanged[str].connect(self.choose_model)
        # 两个按钮水平布局
        btn_handle_predict_widget = QWidget()
        btn_handle_predict_layout = QHBoxLayout()
        btn_handle_predict_layout.addWidget(btn_handle)
        btn_handle_predict_layout.addWidget(btn_predict)
        btn_handle_predict_layout.addWidget(use_model)
        btn_handle_predict_widget.setLayout(btn_handle_predict_layout)
        # 水平布局图片1、2
        img_12_widget = QWidget()
        img_12_widget.setObjectName('border-radius')
        img_12_layout = QHBoxLayout()
        img_1_widget = QWidget()
        img_1_layout = QVBoxLayout()
        img_1_layout.addWidget(img_title1)
        img_1_layout.addWidget(self.img1)
        img_1_layout.addWidget(self.img1_result)
        img_1_layout.addWidget(self.img1_accuracy)
        img_1_widget.setLayout(img_1_layout)
        img_2_widget = QWidget()
        img_2_layout = QVBoxLayout()
        img_2_layout.addWidget(img_title2)
        img_2_layout.addWidget(self.img2)
        img_2_layout.addWidget(self.img2_result)
        img_2_layout.addWidget(self.img2_accuracy)
        img_2_widget.setLayout(img_2_layout)
        img_12_layout.addWidget(img_1_widget)
        img_12_layout.addWidget(img_2_widget)
        img_12_widget.setLayout(img_12_layout)
        # 水平布局图片3、4
        img_34_widget = QWidget()
        img_34_widget.setObjectName('border-radius')
        img_34_layout = QHBoxLayout()
        img_3_widget = QWidget()
        img_3_layout = QVBoxLayout()
        img_3_layout.addWidget(img_title3)
        img_3_layout.addWidget(self.img3)
        img_3_layout.addWidget(self.img3_result)
        img_3_layout.addWidget(self.img3_accuracy)
        img_3_widget.setLayout(img_3_layout)
        img_4_widget = QWidget()
        img_4_layout = QVBoxLayout()
        img_4_layout.addWidget(img_title4)
        img_4_layout.addWidget(self.img4)
        img_4_layout.addWidget(self.img4_result)
        img_4_layout.addWidget(self.img4_accuracy)
        img_4_widget.setLayout(img_4_layout)
        img_34_layout.addWidget(img_3_widget)
        img_34_layout.addWidget(img_4_widget)
        img_34_widget.setLayout(img_34_layout)
        # 完成中间布局
        main_mid_layout.addWidget(mid_title)
        main_mid_layout.addWidget(btn_handle_predict_widget)
        main_mid_layout.addWidget(img_12_widget)
        main_mid_layout.addWidget(img_34_widget)
        main_mid_widget.setLayout(main_mid_layout)

        # 主-中（垂直布局）
        main_right_widget = QWidget()
        main_right_widget.setObjectName('border')
        main_right_layout = QVBoxLayout()
        # 表格标题
        table_title = QLabel('识别记录')
        table_title.setAlignment(Qt.AlignCenter)
        # 识别记录表格
        table_view = QTableView()
        # 设置表格为只读
        table_view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        # 表格标题
        self.table.setHorizontalHeaderLabels(['识别结果', 'CNN模型', '准确率'])
        table_view.setModel(self.table)
        # 导出识别记录按钮
        btn_export = QPushButton('导出详细识别记录')
        btn_export.setObjectName('btn')
        btn_export.clicked.connect(export_record)
        # 花都信息标题
        info_title = QLabel('花朵信息')
        info_title.setAlignment(Qt.AlignCenter)
        # 姓名、学号
        my_info = QLabel('姓名：小伟    学号：3118xxxxxx')
        my_info.setAlignment(Qt.AlignCenter)
        # 完成右侧布局
        main_right_layout.addWidget(table_title)
        main_right_layout.addWidget(table_view)
        main_right_layout.addWidget(btn_export)
        main_right_layout.addWidget(info_title)
        main_right_layout.addWidget(self.flower_info)
        main_right_layout.addWidget(my_info)
        main_right_widget.setLayout(main_right_layout)

        # 完成主布局
        main_layout.addWidget(main_left_widget)
        main_layout.addWidget(main_mid_widget)
        main_layout.addWidget(main_right_widget)
        self.setLayout(main_layout)

        # 宽度高度设置
        self.text_edit.setMaximumHeight(100)
        img_12_widget.setMaximumHeight(399)
        img_34_widget.setMaximumHeight(399)
        self.flower_info.setMaximumHeight(150)
        main_left_widget.setMaximumWidth(422)
        main_mid_widget.setMaximumWidth(654)
        main_right_widget.setMaximumWidth(440)

        # 设置样式
        style = {
            'window': '#window {background-color: #f0f0f0}',
            'border': '#border {border: 2px solid #9498a0}',
            'border-radius': '#border-radius {border: 2px solid #9498a0; border-radius: 15px}',
            'QLabel': 'QLabel {font-size: 18px;font-family: 宋体}',
            'button': '#btn {height: 40px; background-color: #5c94fa; color: white; border-radius: 20px; font-size: 20px;font-family: 黑体}',
            'button:hover': '#btn:hover {background-color: #78c4ec}',
            'button:pressed': '#btn:pressed {background-color:#c44237}'
        }
        all_style = style['button'] + style['button:hover'] + style['button:pressed'] + style['QLabel'] + style['window'] + style['border'] + \
                    style['border-radius']
        self.setStyleSheet(all_style)

    # 本地加载图片
    def load_img(self):
        if self.timer_camera.isActive():
            self.clone_camera()
        # 打开文件选择框选择文件
        openfile_name = QFileDialog.getOpenFileName(self, '选择图片文件', '', 'Image files(*.jpg *.png *.jpeg *.gif *.bmp)')
        # 获取图图片路径
        img = openfile_name[0]
        if img:
            img_show1 = 'resources/temp/to_predict.' + img.split('.')[-1]
            self.main_img = img_show1
            img_show2 = 'resources/temp/to_predict_gray.' + img.split('.')[-1]
            shutil.copy2(img, img_show1)
            img1 = cv2.imread(img_show1)
            img2_gray = cv2.imread(img_show1, cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(img_show2, img2_gray)
            img2 = cv2.imread(img_show2)
            self.img_example.setPixmap(QPixmap(img_show1).scaled(400, 400, Qt.KeepAspectRatio))
            self.img1.setPixmap(QPixmap(img_show1).scaled(280, 280, Qt.KeepAspectRatio))
            self.img2.setPixmap(QPixmap(img_show2).scaled(280, 280, Qt.KeepAspectRatio))
            cv2.imwrite(self.predict, cv2.resize(img1, (224, 224)))
            cv2.imwrite(self.predict_gray, cv2.resize(img2, (224, 224)))
            self.mode = '本地图片'
            self.is_handle = False
            self.img3.setPixmap(QPixmap('resources/img/wait_handle.png').scaled(280, 280, Qt.KeepAspectRatio))
            self.img4.setPixmap(QPixmap('resources/img/wait_handle.png').scaled(280, 280, Qt.KeepAspectRatio))
            self.img1_result.setText('识别结果：')
            self.img1_accuracy.setText('准确率：')
            self.img2_result.setText('识别结果：')
            self.img2_accuracy.setText('准确率：')
            self.img3_result.setText('识别结果：')
            self.img3_accuracy.setText('准确率：')
            self.img4_result.setText('识别结果：')
            self.img4_accuracy.setText('准确率：')
            self.flower_info.setText('')

    # 通过链接或base64加载图片
    def download_img(self):
        if self.timer_camera.isActive():
            self.clone_camera()
        url = self.text_edit.toPlainText()
        if 'http' not in url and 'base64' not in url:
            self.text_edit.setText('请输入正确的花朵图片链接！')
            return print('请输入正确的花朵图片链接！')
        if 'http' in url:
            try:
                img = requests.get(url, timeout=5)
                if not img.headers.get('Content-Type'):
                    suffix = 'jpg'
                else:
                    suffix = img.headers['Content-Type'].split('/')[1]
                    if suffix not in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'JPG', 'JPEG', 'PNG', 'GIF', 'BMP']:
                        self.text_edit.setText(f'检测到非常见的图片格式：{suffix}，请更换链接重试')
                        return print(f'检测到非常见的图片格式：{suffix}，请更换链接重试')
                img_path = f'resources/temp/{int(time.time())}.{suffix}'
                with open(img_path, 'wb') as file:
                    file.write(img.content)
                self.mode = '图片链接'
            except:
                self.text_edit.setText('获取花朵图片失败，请检查网络或更换图片链接！')
                return print('获取花朵图片失败，请检查网络或更换图片链接！')
        else:
            data = re.findall(r'data.+?base64,', url)[0]
            suffix = re.findall(r'/(.+?);base64', data)[0]
            if suffix not in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'JPG', 'JPEG', 'PNG', 'GIF', 'BMP']:
                self.text_edit.setText(f'检测到非常见的图片格式：{suffix}，请更换链接重试')
                return print(f'检测到非常见的图片格式：{suffix}，请更换链接重试')
            try:
                img = base64.b64decode(url.replace(data, ''))
                img_path = f'resources/temp/{int(time.time())}.{suffix}'
                with open(img_path, 'wb') as file:
                    file.write(img)
                self.mode = 'base64'
            except:
                self.text_edit.setText(f'base64编码图片错误，请重新输入')
                return print(f'base64编码图片错误，请重新输入')
        img_show1 = 'resources/temp/to_predict.' + suffix
        self.main_img = img_show1
        img_show2 = 'resources/temp/to_predict_gray.' + suffix
        shutil.copy2(img_path, img_show1)
        img1 = cv2.imread(img_show1)
        img2_gray = cv2.imread(img_show1, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(img_show2, img2_gray)
        img2 = cv2.imread(img_show2)
        self.img_example.setPixmap(QPixmap(img_show1).scaled(400, 400, Qt.KeepAspectRatio))
        self.img1.setPixmap(QPixmap(img_show1).scaled(280, 280, Qt.KeepAspectRatio))
        self.img2.setPixmap(QPixmap(img_show2).scaled(280, 280, Qt.KeepAspectRatio))
        cv2.imwrite(self.predict, cv2.resize(img1, (224, 224)))
        cv2.imwrite(self.predict_gray, cv2.resize(img2, (224, 224)))
        self.is_handle = False
        self.img3.setPixmap(QPixmap('resources/img/wait_handle.png').scaled(280, 280, Qt.KeepAspectRatio))
        self.img4.setPixmap(QPixmap('resources/img/wait_handle.png').scaled(280, 280, Qt.KeepAspectRatio))
        self.img1_result.setText('识别结果：')
        self.img1_accuracy.setText('准确率：')
        self.img2_result.setText('识别结果：')
        self.img2_accuracy.setText('准确率：')
        self.img3_result.setText('识别结果：')
        self.img3_accuracy.setText('准确率：')
        self.img4_result.setText('识别结果：')
        self.img4_accuracy.setText('准确率：')
        self.flower_info.setText('')
        # 删除下载的文件
        os.remove(img_path)

    # 打开摄像头
    def open_camera(self):
        if not self.timer_camera.isActive():
            # 优先加载外置摄像头，如外置摄像头加载失败，则加载自带摄像头
            if not self.cap.open(1):
                self.cap.open(0)
            # 定时器开始计时30ms，结果是每过30ms从摄像头中取一帧显示
            self.timer_camera.start(30)
            # 设置按键内容是关闭摄像头
            self.btn_open_camera.setText('关闭摄像头')
        else:
            self.clone_camera()
            # 关闭摄像头后显示的样式
            self.img_example.setPixmap(QPixmap('resources/img/wait_loading.png').scaled(400, 400, Qt.KeepAspectRatio))

    # 关闭摄像头
    def clone_camera(self):
        # 关闭定时器
        self.timer_camera.stop()
        # 释放视频流
        self.cap.release()
        # 设置按键内容是打开摄像头
        self.btn_open_camera.setText('打开摄像头')

    # 展示视频
    def show_camera(self):
        # 从视频流中读取
        flag, image = self.cap.read()
        # 把读到的帧的大小重新设置为 400*300
        show = cv2.resize(image, (400, 300))
        # 视频色彩转换回RGB
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        # 把读取到的视频数据变成QImage形式
        self.show_image = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
        # 显示QImage
        self.img_example.setPixmap(QPixmap.fromImage(self.show_image))

    # 抓拍图片
    def cap_picture(self):
        if not self.timer_camera.isActive():
            return print('请打开摄像头再进行抓拍！')
        try:
            img_show1 = 'resources/temp/to_predict.png'
            self.main_img = img_show1
            img_show2 = 'resources/temp/to_predict_gray.png'
            self.show_image.save(img_show1, 'PNG')
            self.clone_camera()
            # 展示抓拍的图片
            self.img_example.setPixmap(QPixmap.fromImage(self.show_image))
            img1 = cv2.imread(img_show1)
            img2_gray = cv2.imread(img_show1, cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(img_show2, img2_gray)
            img2 = cv2.imread(img_show2)
            self.img_example.setPixmap(QPixmap(img_show1).scaled(400, 400, Qt.KeepAspectRatio))
            self.img1.setPixmap(QPixmap(img_show1).scaled(280, 280, Qt.KeepAspectRatio))
            self.img2.setPixmap(QPixmap(img_show2).scaled(280, 280, Qt.KeepAspectRatio))
            cv2.imwrite(self.predict, cv2.resize(img1, (224, 224)))
            cv2.imwrite(self.predict_gray, cv2.resize(img2, (224, 224)))
            self.mode = '摄像头抓拍'
            self.is_handle = False
            self.img3.setPixmap(QPixmap('resources/img/wait_handle.png').scaled(280, 280, Qt.KeepAspectRatio))
            self.img4.setPixmap(QPixmap('resources/img/wait_handle.png').scaled(280, 280, Qt.KeepAspectRatio))
            self.img1_result.setText('识别结果：')
            self.img1_accuracy.setText('准确率：')
            self.img2_result.setText('识别结果：')
            self.img2_accuracy.setText('准确率：')
            self.img3_result.setText('识别结果：')
            self.img3_accuracy.setText('准确率：')
            self.img4_result.setText('识别结果：')
            self.img4_accuracy.setText('准确率：')
            self.flower_info.setText('')
            print('抓拍成功！')
        except:
            print('抓拍失败，请先打开摄像头再进行抓拍！')

    def handle(self):
        if not self.main_img:
            return print('请先加载图片再进行图片处理')
        global rect, left_button_down, left_button_up, x_min, y_min, x_max, y_max
        # 初始坐标和鼠标左键状态
        rect = [0, 0, 0, 0]
        left_button_down = False
        left_button_up = True
        # 加载图片并显示图片，设置鼠标响应回调函数
        img = cv2.imread(self.main_img)
        cv2.imshow('img', img)

        # 鼠标事件
        def on_mouse(event, x, y, flag, param):
            global rect, left_button_down, left_button_up, x_min, y_min, x_max, y_max
            # 鼠标左键点击
            if event == cv2.EVENT_LBUTTONDOWN:
                rect[0] = rect[2] = x
                rect[1] = rect[3] = y
                left_button_down = True
                left_button_up = False

            # 鼠标移动事件
            if event == cv2.EVENT_MOUSEMOVE:
                if left_button_down and not left_button_up:
                    rect[2] = x
                    rect[3] = y

            # 鼠标左键松开
            if event == cv2.EVENT_LBUTTONUP:
                if left_button_down and not left_button_up:
                    x_min = min(rect[0], rect[2])
                    y_min = min(rect[1], rect[3])
                    x_max = max(rect[0], rect[2])
                    y_max = max(rect[1], rect[3])
                    rect[0] = x_min
                    rect[1] = y_min
                    rect[2] = x_max
                    rect[3] = y_max
                    left_button_down = False
                    left_button_up = True

        cv2.setMouseCallback('img', on_mouse)
        # 每10毫秒刷新一次
        while cv2.waitKey(10) == -1:
            # 左键按下，画矩阵
            if left_button_down and not left_button_up:
                img_copy = img.copy()
                # 图片，长方形框左上角坐标, 长方形框右下角坐标， 边框颜色，边框线大小
                cv2.rectangle(img_copy, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 2)
                cv2.imshow('img', img_copy)
            # 左键松开，矩形画好
            if not left_button_down and left_button_up and rect[2] - rect[0] != 0 and rect[3] - rect[1] != 0:
                img_cut = img[rect[1]:rect[3], rect[0]:rect[2]]
                cv2.imwrite('resources/temp/to_predict_cut.png', img_cut)
                cv2.imwrite(self.predict_cut, cv2.resize(img_cut, (224, 224)))
                self.img3.setPixmap(QPixmap('resources/temp/to_predict_cut.png').scaled(280, 280, Qt.KeepAspectRatio))
                # 转换为宽度高度
                rect[2] = rect[2] - rect[0]
                rect[3] = rect[3] - rect[1]
                # 原图mask、输入的单通道图像
                mask = np.zeros(img.shape[:2], np.uint8)
                # 临时背景模型数组
                bgd_model = np.zeros((1, 65), np.float64)
                # 临时前景模型数组
                fgd_model = np.zeros((1, 65), np.float64)
                # Grabcut图像分割
                cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
                # 重新初始化坐标
                rect = [0, 0, 0, 0]
                mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
                img_show = img * mask2[:, :, np.newaxis]
                # 背景换白色
                img_show[np.where((img_show == [0, 0, 0]).all(axis=2))] = [255, 255, 255]
                # 输出白色背景
                img_show = img_show[y_min:y_max, x_min:x_max]
                # 输出裁剪后白色背景图
                cv2.imwrite('resources/temp/to_predict_cut_prospects.png', img_show)
                cv2.imwrite(self.predict_cut_prospects, cv2.resize(img_show, (224, 224)))
                self.img4.setPixmap(QPixmap('resources/temp/to_predict_cut_prospects.png').scaled(280, 280, Qt.KeepAspectRatio))
                self.is_handle = True
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 花朵识别
    def predict_img(self):
        info_list = []

        def mode1():
            # 原图
            predict_img = Image.open(self.predict)
            predict_img_arr = np.asarray(predict_img)
            predict_img_output = self.model(predict_img_arr.reshape([1, 224, 224, 3]))
            predict_img_index = int(np.argmax(predict_img_output))
            # 获得对应的花朵名称
            predict_img_result, predict_img_information = get_info(predict_img_index)
            predict_img_acc = '%.2f' % (np.max(predict_img_output) * 100)
            self.img1_result.setText(f'识别结果:{predict_img_result}')
            self.img1_accuracy.setText(f'准确率:{predict_img_acc}%')
            info_list.append([float(predict_img_acc), predict_img_result, predict_img_information])
            # 灰度图
            predict_gray_img = Image.open(self.predict_gray)
            predict_gray_img_arr = np.asarray(predict_gray_img)
            predict_gray_img_output = self.model(predict_gray_img_arr.reshape([1, 224, 224, 3]))
            predict_gray_img_index = int(np.argmax(predict_gray_img_output))
            # 获得对应的花朵名称
            predict_gray_img_result, predict_gray_img_information = get_info(predict_gray_img_index)
            predict_gray_img_acc = '%.2f' % (np.max(predict_gray_img_output) * 100)
            self.img2_result.setText(f'识别结果:{predict_gray_img_result}')
            self.img2_accuracy.setText(f'准确率:{predict_gray_img_acc}%')
            info_list.append([float(predict_gray_img_acc), predict_gray_img_result, predict_gray_img_information])

        def mode2():
            # 裁剪图
            predict_cut_img = Image.open(self.predict_cut)
            predict_cut_img_arr = np.asarray(predict_cut_img)
            predict_cut_img_output = self.model(predict_cut_img_arr.reshape([1, 224, 224, 3]))
            predict_cut_img_index = int(np.argmax(predict_cut_img_output))
            # 获得对应的花朵名称
            predict_cut_img_result, predict_img_information = get_info(predict_cut_img_index)
            predict_cut_img_acc = '%.2f' % (np.max(predict_cut_img_output) * 100)
            self.img3_result.setText(f'识别结果:{predict_cut_img_result}')
            self.img3_accuracy.setText(f'准确率:{predict_cut_img_acc}%')
            info_list.append([float(predict_cut_img_acc), predict_cut_img_result, predict_img_information])
            # 灰度图
            predict_cut_prospects_img = Image.open(self.predict_cut_prospects)
            predict_cut_prospects_img_arr = np.asarray(predict_cut_prospects_img)
            predict_cut_prospects_img_output = self.model(predict_cut_prospects_img_arr.reshape([1, 224, 224, 3]))
            predict_cut_prospects_img_index = int(np.argmax(predict_cut_prospects_img_output))
            # 获得对应的花朵名称
            predict_cut_prospects_img_result, predict_cut_prospects_img_information = get_info(predict_cut_prospects_img_index)
            predict_cut_prospects_img_acc = '%.2f' % (np.max(predict_cut_prospects_img_output) * 100)
            self.img4_result.setText(f'识别结果:{predict_cut_prospects_img_result}')
            self.img4_accuracy.setText(f'准确率:{predict_cut_prospects_img_acc}%')
            info_list.append([float(predict_cut_prospects_img_acc), predict_cut_prospects_img_result, predict_cut_prospects_img_information])

        if not self.main_img:
            return print('请先加载图片再进行花朵识别')
        if self.is_handle:
            mode1()
            mode2()
        else:
            mode1()
        info_list.sort(reverse=True)
        self.flower_info.setText(info_list[0][2])
        self.record_num += 1
        simple_data = [info_list[0][1], self.my_model, f'{info_list[0][0]}%']
        print(simple_data + [self.mode])
        # 向表格中插入数据并居中显示
        for i, info in enumerate(simple_data):
            self.table.setItem(self.record_num - 1, i, QStandardItem(info))
            self.table.item(self.record_num - 1, i).setTextAlignment(Qt.AlignCenter)

        # 向数据库中写入识别记录
        data = (self.record_num, info_list[0][1], self.my_model, f'{info_list[0][0]}%', self.mode, get_time(), info_list[0][2])
        write_record(data)

    # 选择模型
    def choose_model(self, model):
        self.model = tf.keras.models.load_model(f'resources/models/{model}.h5')
        self.my_model = model
        print(f'当前选择的模型是：{model}')

    # 界面关闭事件，询问用户是否关闭
    def closeEvent(self, event):
        choose = QMessageBox.question(self, '退出', '是否要退出程序？', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if choose == QMessageBox.Yes:
            print('退出花朵识别系统！')
            # 尝试清除resources/temp里面的临时文件
            for i in os.listdir('resources/temp'):
                try:
                    os.remove('resources/temp/' + i)
                except:
                    pass
            event.accept()
        else:
            event.ignore()


# 格式化输出当前时间
def get_time():
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return now


# 从数据库中读取花朵类名和花朵信息
def get_info(index):
    sql1 = f'select * from labels where id = ?'
    name = check(sql1, (index,))[0][1]
    sql2 = f'select * from informations where name = ?'
    information = check(sql2, (name,))[0][2]
    return name, information


# 向数据库写入识别记录
def write_record(data):
    sql = 'insert into records (id, name, model, accuracy, source, time, info) values (?, ?, ?, ?, ?, ?, ?)'
    insert(sql, data)


# 导出识别记录
def export_record():
    msg = save_excel()
    msg_box = QMessageBox(QMessageBox.Information, '导出识别记录', msg)
    msg_box.setObjectName('message')
    msg_box.exec_()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    x = MainWindow()
    x.show()
    sys.exit(app.exec_())
