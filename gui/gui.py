import sys
# from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QStackedWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox, QFileDialog
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal

from pyaet.main_reconstruction1 import main_reconstruction
from pyaet.main_polynomial_tracing2 import main_polynomial_tracing
from pyaet.main_classification3 import main_classification
from pyaet.main_position_refinement4 import main_position_refinement


class MyApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # 创建一个堆叠小部件
        self.stack = QStackedWidget(self)
        
        # 创建五个页面，并在其中添加不同的内容
        #####################
        # Home page
        self.home_page = QWidget()
        self.home_layout = QVBoxLayout()

        image_label = QLabel()
        # image_label.setFixedWidth(800)  # 设置 QLabel 的固定宽度
        # image_label.setFixedHeight(300)  # 设置 QLabel 的固定高度
        pixmap = QPixmap('pics/main_aet.png')
        image_label.setPixmap(pixmap.scaled(800, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        # scaled_pixmap = pixmap.scaled(image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        # image_label.setPixmap(pixmap)
        image_label.setAlignment(Qt.AlignCenter)  # 水平和垂直居中
        self.home_layout.addWidget(image_label)

        text_label = QLabel("J.  Miao,  P.  Ercius,  and S.  J.  L.  Billinge,  Atomic electron tomography: 3D structures without crystals,  Science,  353(6306),  aaf2157 (2016).")
        text_label.setAlignment(Qt.AlignCenter)  # 同样可以设置文本居中
        text_label.setWordWrap(True)  # 开启自动换行
        self.home_layout.addWidget(text_label)

        self.home_page.setLayout(self.home_layout)
        self.stack.addWidget(self.home_page)
        #####################

        #####################
        # Reconstruction page
        self.recon_page = QWidget()
        self.recon_layout = QVBoxLayout()

        # 添加文件选择功能
        recon_pj_file_layout = QHBoxLayout()
        recon_pj_file_label = QLabel('Input Projection File Path:')
        recon_pj_file_edit = QLineEdit()
        recon_pj_file_button = QPushButton('Browse')
        recon_pj_file_button.clicked.connect(lambda _, fe=recon_pj_file_edit: self.select_file(fe))
        # recon_pj_file_button.clicked.connect(self.select_file)
        recon_pj_file_layout.addWidget(recon_pj_file_label)
        recon_pj_file_layout.addWidget(recon_pj_file_edit)
        recon_pj_file_layout.addWidget(recon_pj_file_button)
        self.recon_layout.addLayout(recon_pj_file_layout)

        # 添加文件选择功能
        recon_angle_file_layout = QHBoxLayout()
        recon_angle_file_label = QLabel('Input Angle File Path:', self.recon_page)
        recon_angle_file_edit = QLineEdit(self.recon_page)
        recon_angle_file_button = QPushButton('Browse', self.recon_page)
        recon_angle_file_button.clicked.connect(lambda _, fe=recon_angle_file_edit: self.select_file(fe))
        recon_angle_file_layout.addWidget(recon_angle_file_label)
        recon_angle_file_layout.addWidget(recon_angle_file_edit)
        recon_angle_file_layout.addWidget(recon_angle_file_button)
        self.recon_layout.addLayout(recon_angle_file_layout)

        # 文本框
        recon_param_oversampling_layout = QHBoxLayout()
        recon_param_oversampling_label = QLabel(f'Oversampling Ratio:')
        recon_param_oversampling_edit = QLineEdit('3')
        recon_param_oversampling_layout.addWidget(recon_param_oversampling_label)
        recon_param_oversampling_layout.addWidget(recon_param_oversampling_edit)
        self.recon_layout.addLayout(recon_param_oversampling_layout)

        # 文本框
        recon_param_iteration_layout = QHBoxLayout()
        recon_param_iteration_label = QLabel(f'Number of Iterations:')
        recon_param_iteration_edit = QLineEdit('100')
        recon_param_iteration_layout.addWidget(recon_param_iteration_label)
        recon_param_iteration_layout.addWidget(recon_param_iteration_edit)
        self.recon_layout.addLayout(recon_param_iteration_layout)

        # 下拉列表
        recon_param_parallel_layout = QHBoxLayout()
        recon_param_parallel_label = QLabel('Parallel Computation:')
        recon_param_parallel_combo = QComboBox()
        recon_param_parallel_combo.addItems(['True', 'False'])
        recon_param_parallel_layout.addWidget(recon_param_parallel_label)
        recon_param_parallel_layout.addWidget(recon_param_parallel_combo)
        self.recon_layout.addLayout(recon_param_parallel_layout)

        # 文本框
        recon_output_layout = QHBoxLayout()
        recon_output_label = QLabel(f'Output filename:')
        recon_output_edit = QLineEdit('output_reconstruction')
        recon_output_layout.addWidget(recon_output_label)
        recon_output_layout.addWidget(recon_output_edit)
        self.recon_layout.addLayout(recon_output_layout)

        # 添加执行操作的按钮
        recon_run_button = QPushButton('Run1')
        # recon_run_button.clicked.connect(
        #     lambda _,
        #            recon_pj_file_path = recon_pj_file_edit.text(),
        #            recon_angle_file_path = recon_angle_file_edit.text():
        #     self.runPythonCode1(recon_pj_file_path, recon_angle_file_path))
        recon_run_button.clicked.connect(
            lambda: self.run_main_reconstruction(recon_pj_file_edit.text(),
                                                 recon_angle_file_edit.text(),
                                                 recon_param_oversampling_edit.text(),
                                                 recon_param_iteration_edit.text(),
                                                 recon_param_parallel_combo.currentText(),
                                                 recon_output_edit.text())
        )
        self.recon_layout.addWidget(recon_run_button)

        # main_reconstruction(pj_filename, angle_filename, resire_param, results_filename, output_fn)

        self.recon_page.setLayout(self.recon_layout)
        self.stack.addWidget(self.recon_page)
        #####################


        #####################
        # Tracing page
        self.tracing_page = QWidget()
        self.tracing_layout = QVBoxLayout()

        # 添加文件选择功能
        tracing_recon_file_layout = QHBoxLayout()
        tracing_recon_file_label = QLabel('Input Reconstruction File Path:')
        tracing_recon_file_edit = QLineEdit()
        tracing_recon_file_button = QPushButton('Browse')
        tracing_recon_file_button.clicked.connect(lambda _, fe=tracing_recon_file_edit: self.select_file(fe))
        tracing_recon_file_layout.addWidget(tracing_recon_file_label)
        tracing_recon_file_layout.addWidget(tracing_recon_file_edit)
        tracing_recon_file_layout.addWidget(tracing_recon_file_button)
        self.tracing_layout.addLayout(tracing_recon_file_layout)

        # 添加执行操作的按钮
        tracing_run_button = QPushButton('Run2')
        tracing_run_button.clicked.connect(
            lambda _,
                   tracing_recon_file_path = tracing_recon_file_edit.text():
            self.runPythonCode(tracing_recon_file_path))
        self.tracing_layout.addWidget(tracing_run_button)

        self.tracing_page.setLayout(self.tracing_layout)
        self.stack.addWidget(self.tracing_page)
        #####################


        #####################
        # Classification page
        self.class_page = QWidget()
        self.class_layout = QVBoxLayout()

        # 添加文件选择功能
        class_recon_file_layout = QHBoxLayout()
        class_recon_file_label = QLabel('Input Reconstruction File Path:', self.class_page)
        class_recon_file_edit = QLineEdit(self.class_page)
        class_recon_file_button = QPushButton('Browse', self.class_page)
        class_recon_file_button.clicked.connect(lambda _, fe=class_recon_file_edit: self.select_file(fe))
        class_recon_file_layout.addWidget(class_recon_file_label)
        class_recon_file_layout.addWidget(class_recon_file_edit)
        class_recon_file_layout.addWidget(class_recon_file_button)
        self.class_layout.addLayout(class_recon_file_layout)

        # 添加文件选择功能
        class_model_file_layout = QHBoxLayout()
        class_model_file_label = QLabel('Input Traced Model File Path:', self.class_page)
        class_model_file_edit = QLineEdit(self.class_page)
        class_model_file_button = QPushButton('Browse', self.class_page)
        class_model_file_button.clicked.connect(lambda _, fe=class_model_file_edit: self.select_file(fe))
        class_model_file_layout.addWidget(class_model_file_label)
        class_model_file_layout.addWidget(class_model_file_edit)
        class_model_file_layout.addWidget(class_model_file_button)
        self.class_layout.addLayout(class_model_file_layout)

        # 添加执行操作的按钮
        class_run_button = QPushButton('Run3')
        class_run_button.clicked.connect(
            lambda _,
                   class_recon_file_path = class_recon_file_edit.text(),
                   class_model_file_path = class_model_file_edit.text():
            self.runPythonCode(class_recon_file_path, class_model_file_path))
        self.class_layout.addWidget(class_run_button)

        self.class_page.setLayout(self.class_layout)
        self.stack.addWidget(self.class_page)
        #####################

        #####################
        # Position Refinement page
        self.refine_page = QWidget()
        self.refine_layout = QVBoxLayout()

        # 添加文件选择功能
        refine_pj_file_layout = QHBoxLayout()
        refine_pj_file_label = QLabel('Input Projection File Path:')
        refine_pj_file_edit = QLineEdit()
        refine_pj_file_button = QPushButton('Browse')
        refine_pj_file_button.clicked.connect(lambda _, fe=refine_pj_file_edit: self.select_file(fe))
        refine_pj_file_layout.addWidget(refine_pj_file_label)
        refine_pj_file_layout.addWidget(refine_pj_file_edit)
        refine_pj_file_layout.addWidget(refine_pj_file_button)
        self.refine_layout.addLayout(refine_pj_file_layout)

        # 添加文件选择功能
        refine_angle_file_layout = QHBoxLayout()
        refine_angle_file_label = QLabel('Input Angle File Path:')
        refine_angle_file_edit = QLineEdit()
        refine_angle_file_button = QPushButton('Browse')
        refine_angle_file_button.clicked.connect(lambda _, fe=refine_angle_file_edit: self.select_file(fe))
        refine_angle_file_layout.addWidget(refine_angle_file_label)
        refine_angle_file_layout.addWidget(refine_angle_file_edit)
        refine_angle_file_layout.addWidget(refine_angle_file_button)
        self.refine_layout.addLayout(refine_angle_file_layout)

        # 添加文件选择功能
        refine_model_file_layout = QHBoxLayout()
        refine_model_file_label = QLabel('Input Traced Model File Path:')
        refine_model_file_edit = QLineEdit()
        refine_model_file_button = QPushButton('Browse')
        refine_model_file_button.clicked.connect(lambda _, fe=refine_model_file_edit: self.select_file(fe))
        refine_model_file_layout.addWidget(refine_model_file_label)
        refine_model_file_layout.addWidget(refine_model_file_edit)
        refine_model_file_layout.addWidget(refine_model_file_button)
        self.refine_layout.addLayout(refine_model_file_layout)

        # 添加文件选择功能
        refine_atom_file_layout = QHBoxLayout()
        refine_atom_file_label = QLabel('Input Traced atom File Path:')
        refine_atom_file_edit = QLineEdit()
        refine_atom_file_button = QPushButton('Browse')
        refine_atom_file_button.clicked.connect(lambda _, fe=refine_atom_file_edit: self.select_file(fe))
        refine_atom_file_layout.addWidget(refine_atom_file_label)
        refine_atom_file_layout.addWidget(refine_atom_file_edit)
        refine_atom_file_layout.addWidget(refine_atom_file_button)
        self.refine_layout.addLayout(refine_atom_file_layout)

        # 文本框
        refine_output_layout = QHBoxLayout()
        refine_output_label = QLabel(f'Output filename:')
        refine_output_edit = QLineEdit('output_model')
        refine_output_layout.addWidget(refine_output_label)
        refine_output_layout.addWidget(refine_output_edit)
        self.refine_layout.addLayout(refine_output_layout)

        # 添加执行操作的按钮
        refine_run_button = QPushButton('Run4')
        refine_run_button.clicked.connect(
            lambda _, refine_pj_file_path = refine_pj_file_edit.text(),
                   refine_angle_file_path=refine_angle_file_edit.text(),
                   refine_model_file_path = refine_model_file_edit.text(),
                   refine_atom_file_path = refine_atom_file_edit.text(),
                   refine_output_file_name = refine_output_edit.text():
            self.run_main_position_refinement(refine_pj_file_path, refine_angle_file_path, refine_model_file_path, refine_atom_file_path, refine_output_file_name))
        self.refine_layout.addWidget(refine_run_button)

        self.refine_page.setLayout(self.refine_layout)
        self.stack.addWidget(self.refine_page)
        #####################


        # 创建五个按钮，并设置它们的clicked信号和布局
        self.buttons = []
        self.button_layout = QHBoxLayout()
        for i in range(5):
            if i == 0:
                button = QPushButton(f'Home', self)
            if i == 1:
                button = QPushButton(f'Reconstruction', self)
            if i == 2:
                button = QPushButton(f'Tracing', self)
            if i == 3:
                button = QPushButton(f'Classification', self)
            if i == 4:
                button = QPushButton(f'Position Refinement', self)
            button.clicked.connect(lambda _, b=i: self.display_page(b))
            self.buttons.append(button)
            self.button_layout.addWidget(button)


        # 设置窗口的总布局
        self.main_layout = QVBoxLayout()
        self.main_layout.addLayout(self.button_layout)
        self.main_layout.addWidget(self.stack)

        # 设置中心窗口的布局
        self.central_widget = QWidget()
        self.central_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.central_widget)

        # 设置窗口的初始位置和大小
        self.setGeometry(200, 200, 800, 600)
        self.setWindowTitle('Atomic Electron Tomography')



    # 显示指定索引页面的方法，并更改按钮颜色
    def display_page(self, index):
        self.stack.setCurrentIndex(index)
        print("self.stack.setCurrentIndex(index)")
        print(index)
        # 更新所有按钮的颜色
        for i, button in enumerate(self.buttons):
            if i == index:
                button.setStyleSheet("background-color: lightblue")  # 选中的按钮更改颜色
            else:
                button.setStyleSheet("")  # 其他按钮恢复默认颜色

    # 文件选择对话框
    # def select_file(self, line_edit):
    #     file_name, _ = QFileDialog.getOpenFileName(self, "Select File")
    #     if file_name:  # 确保用户选择了文件
    #         line_edit.setText(file_name)  # 更新文本框内容
    #         print(file_name)

    def select_file(self, line_edit):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "Numpy Files (*.npy);;Matlab Files (*.mat)")
        if file_path:  # 确保用户选择了文件
            line_edit.setText(file_path)  # 更新文本框内容
            print(file_path)

    # 运行简单的Python代码
    def runPythonCode(self, name):
        # 这里可以替换为任何想要执行的Python代码
        print(f"Executing Python code for {name}")

    def run_main_reconstruction(self, pj_filename, angle_filename, resire_param_oversampling, resire_param_iteration, resire_param_parallel, output_fn):
        resire_param = {
            "oversampling_ratio": 3,
            "num_iterations": 10,
            "monitor_R": True,
            "monitorR_loopLength": 2,
            "gridding_method": 1,
            "vector3": [1, 0, 0],
            "use_parallel": True,
            "save_temp": True,
            "save_loopLength": 5
        }
        resire_param['oversampling_ratio'] = int(float(resire_param_oversampling))
        resire_param['num_iterations'] = int(float(resire_param_iteration))
        if resire_param_parallel == "True":
            resire_param['use_parallel'] = True
        if resire_param_parallel == "False":
            resire_param['use_parallel'] = False

        main_reconstruction(pj_filename, angle_filename, resire_param, output_fn)


    def run_main_position_refinement(self, projections_file_path, angles_file_path, model_file_path, atoms_file_path, output_fn):
        print("now this line")
        print(projections_file_path)
        print("end now this line")
        print(type(projections_file_path))
        print(type(output_fn))
        main_position_refinement(projections_file_path, angles_file_path, model_file_path, atoms_file_path, output_fn)
        return



if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    ex.show()
    sys.exit(app.exec_())
