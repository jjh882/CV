import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QTextEdit
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image


class AnimalRecognitionSystem(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.model = MobileNetV2(weights='imagenet')

    def initUI(self):
        # 设置窗口标题
        self.setWindowTitle('动物识别专家系统')
        self.resize(980, 450)

        # 创建布局
        main_layout = QVBoxLayout()

        # 创建并添加标题标签
        title_label = QLabel('动物识别专家系统', self)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        # 创建并添加图片展示区域
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setPixmap(QPixmap().scaled(200, 200, Qt.KeepAspectRatio))
        main_layout.addWidget(self.image_label)

        # 创建并添加结果展示区域
        self.result_text = QTextEdit(self)
        self.result_text.setReadOnly(True)
        main_layout.addWidget(self.result_text)

        # 创建并添加按钮
        button_layout = QHBoxLayout()

        self.upload_button = QPushButton('上传图片', self)
        self.upload_button.clicked.connect(self.upload_image)
        button_layout.addWidget(self.upload_button)

        self.recognize_button = QPushButton('识别动物', self)
        self.recognize_button.clicked.connect(self.recognize_animal)
        button_layout.addWidget(self.recognize_button)

        main_layout.addLayout(button_layout)

        # 设置主布局
        self.setLayout(main_layout)

    def upload_image(self):
        # 打开文件对话框选择图片
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "图片文件 (*.jpg *.jpeg *.png);;所有文件 (*)",
                                                   options=options)
        if file_name:
            self.image_path = file_name
            self.image_label.setPixmap(QPixmap(file_name).scaled(200, 200, Qt.KeepAspectRatio))

    def recognize_animal(self):
        if hasattr(self, 'image_path'):
            img = image.load_img(self.image_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            predictions = self.model.predict(img_array)
            decoded_predictions = decode_predictions(predictions, top=1)[0][0]
            result = f'识别结果：{decoded_predictions[1]}，概率：{decoded_predictions[2]:.2f}'
            self.result_text.setText(result)
        else:
            self.result_text.setText('请先上传图片。')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AnimalRecognitionSystem()
    ex.show()
    sys.exit(app.exec_())