import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QStackedWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox, QFileDialog

class MyApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # 创建一个堆叠小部件
        self.stack = QStackedWidget(self)
        
        # 创建五个页面，并在其中添加不同的内容
        self.pages = []
        self.phase_names = ['Ferrite', 'Austenite', 'Martensite', 'Cementite', 'Pearlite']
        self.additional_names = ['Microstructure', 'Elasticity', 'Fracture', 'Alloy Composition', 'Thermal Conductivity']
        self.material_properties = ['Hardness', 'Toughness', 'Ductility', 'Malleability', 'Elasticity']
        for i in range(5):
            if i==0:
                page = QWidget()
                layout = QVBoxLayout()

                page.setLayout(layout)
                self.pages.append(page)
                self.stack.addWidget(page)
            else:
                page = QWidget()
                layout = QVBoxLayout()

                # 添加文件选择功能
                file_layout = QHBoxLayout()
                file_label = QLabel('Projection File Path:', page)
                file_edit = QLineEdit(page)
                file_button = QPushButton('Browse', page)
                file_button.clicked.connect(lambda _, fe=file_edit: self.selectFile(fe))
                file_layout.addWidget(file_label)
                file_layout.addWidget(file_edit)
                file_layout.addWidget(file_button)
                layout.addLayout(file_layout)

                # 添加文件选择功能
                file_layout = QHBoxLayout()
                file_label = QLabel('Angle File Path:', page)
                file_edit = QLineEdit(page)
                file_button = QPushButton('Browse', page)
                file_button.clicked.connect(lambda _, fe=file_edit: self.selectFile(fe))
                file_layout.addWidget(file_label)
                file_layout.addWidget(file_edit)
                file_layout.addWidget(file_button)
                layout.addLayout(file_layout)

                # 相名称和文本框
                phase_layout = QHBoxLayout()
                phase_label = QLabel(f'Oversampling Ratio:', page)
                phase_edit = QLineEdit('3', page)
                phase_layout.addWidget(phase_label)
                phase_layout.addWidget(phase_edit)
                layout.addLayout(phase_layout)

                # 额外的文本框
                additional_layout = QHBoxLayout()
                additional_label = QLabel(f'Number of Iterations:', page)
                additional_edit = QLineEdit('100', page)
                additional_layout.addWidget(additional_label)
                additional_layout.addWidget(additional_edit)
                layout.addLayout(additional_layout)

                # 下拉列表
                property_layout = QHBoxLayout()
                property_label = QLabel('Parallel Computation:', page)
                property_combo = QComboBox(page)
                property_combo.addItems(['True', 'False'])
                property_layout.addWidget(property_label)
                property_layout.addWidget(property_combo)
                layout.addLayout(property_layout)

                # 添加执行操作的按钮
                action_button = QPushButton('Run', page)
                action_button.clicked.connect(lambda _, name=self.phase_names[i % len(self.phase_names)]: self.runPythonCode(name))
                layout.addWidget(action_button)


                page.setLayout(layout)
                self.pages.append(page)
                self.stack.addWidget(page)

        # 创建五个按钮，并设置它们的clicked信号和布局
        self.buttons = []
        self.buttonLayout = QHBoxLayout()
        for i in range(5):
            if i==0:
                button = QPushButton(f'Home', self)
                button.clicked.connect(lambda _, b=i: self.displayPage(b))
                self.buttons.append(button)
                self.buttonLayout.addWidget(button)
            if i==1:
                button = QPushButton(f'Reconstruction', self)
                button.clicked.connect(lambda _, b=i: self.displayPage(b))
                self.buttons.append(button)
                self.buttonLayout.addWidget(button)
            if i==2:
                button = QPushButton(f'Tracing', self)
                button.clicked.connect(lambda _, b=i: self.displayPage(b))
                self.buttons.append(button)
                self.buttonLayout.addWidget(button)
            if i==3:
                button = QPushButton(f'Classification', self)
                button.clicked.connect(lambda _, b=i: self.displayPage(b))
                self.buttons.append(button)
                self.buttonLayout.addWidget(button)
            if i==4:
                button = QPushButton(f'Position Refinement', self)
                button.clicked.connect(lambda _, b=i: self.displayPage(b))
                self.buttons.append(button)
                self.buttonLayout.addWidget(button)
        # 设置窗口的总布局
        self.mainLayout = QVBoxLayout()
        self.mainLayout.addLayout(self.buttonLayout)
        self.mainLayout.addWidget(self.stack)

        # 设置中心窗口的布局
        self.centralWidget = QWidget()
        self.centralWidget.setLayout(self.mainLayout)
        self.setCentralWidget(self.centralWidget)

        # 设置窗口的初始位置和大小
        self.setGeometry(200, 200, 800, 600)
        self.setWindowTitle('Atomic Electron Tomography')

    # 显示指定索引页面的方法，并更改按钮颜色
    def displayPage(self, index):
        self.stack.setCurrentIndex(index)
        # 更新所有按钮的颜色
        for i, button in enumerate(self.buttons):
            if i == index:
                button.setStyleSheet("background-color: lightblue")  # 选中的按钮更改颜色
            else:
                button.setStyleSheet("")  # 其他按钮恢复默认颜色

    # 文件选择对话框
    def selectFile(self, line_edit):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select File")
        if file_name:  # 确保用户选择了文件
            line_edit.setText(file_name)  # 更新文本框内容

    # 运行简单的Python代码
    def runPythonCode(self, name):
        # 这里可以替换为任何想要执行的Python代码
        print(f"Executing Python code for {name}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    ex.show()
    sys.exit(app.exec_())
