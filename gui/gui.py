import sys
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from scipy.stats import gaussian_kde

from pyaet.main_reconstruction1 import main_reconstruction
from pyaet.main_polynomial_tracing2 import main_polynomial_tracing
from pyaet.main_classification3 import main_classification
from pyaet.main_position_refinement4 import main_position_refinement
from pyaet.analysis.calc_pdf import calc_pdf
from pyaet.analysis.calc_boo import calc_boo
from pyaet.analysis.calc_csro import calc_csro
from pyaet.analysis.calc_voronoi import calc_voronoi
import warnings

# 抑制sipPyTypeDict弃用警告
warnings.filterwarnings("ignore", category=DeprecationWarning, message="sipPyTypeDict")

class MyApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # create stacking widgets.
        self.stack = QStackedWidget(self)
        
        # Create pages one by one.
        #####################
        # Home page
        self.home_page = QWidget()
        self.home_layout = QVBoxLayout()

        image_label = QLabel()
        pixmap = QPixmap('pics/main_aet.png')
        image_label.setPixmap(pixmap.scaled(800, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        image_label.setAlignment(Qt.AlignCenter)  # Vertical and Horizontal Center
        self.home_layout.addWidget(image_label)

        text_label = QLabel("J.  Miao,  P.  Ercius,  and S.  J.  L.  Billinge,  Atomic electron tomography: 3D structures without crystals,  Science,  353(6306),  aaf2157 (2016).")
        text_label.setAlignment(Qt.AlignCenter)  # Text center
        text_label.setWordWrap(True)  # Auto new line
        self.home_layout.addWidget(text_label)

        self.home_page.setLayout(self.home_layout)
        self.stack.addWidget(self.home_page)
        #####################

        #####################
        # Reconstruction page
        self.recon_page = QWidget()
        self.recon_layout = QVBoxLayout()

        recon_pj_file_layout = QHBoxLayout()
        recon_pj_file_label = QLabel('Input Projection File Path:')
        recon_pj_file_edit = QLineEdit()
        recon_pj_file_button = QPushButton('Browse')
        recon_pj_file_button.clicked.connect(lambda _, fe=recon_pj_file_edit: self.select_file(fe))
        recon_pj_file_layout.addWidget(recon_pj_file_label)
        recon_pj_file_layout.addWidget(recon_pj_file_edit)
        recon_pj_file_layout.addWidget(recon_pj_file_button)
        self.recon_layout.addLayout(recon_pj_file_layout)

        recon_angle_file_layout = QHBoxLayout()
        recon_angle_file_label = QLabel('Input Angle File Path:', self.recon_page)
        recon_angle_file_edit = QLineEdit(self.recon_page)
        recon_angle_file_button = QPushButton('Browse', self.recon_page)
        recon_angle_file_button.clicked.connect(lambda _, fe=recon_angle_file_edit: self.select_file(fe))
        recon_angle_file_layout.addWidget(recon_angle_file_label)
        recon_angle_file_layout.addWidget(recon_angle_file_edit)
        recon_angle_file_layout.addWidget(recon_angle_file_button)
        self.recon_layout.addLayout(recon_angle_file_layout)

        recon_param_oversampling_layout = QHBoxLayout()
        recon_param_oversampling_label = QLabel(f'Oversampling Ratio:')
        recon_param_oversampling_edit = QLineEdit('3')
        recon_param_oversampling_layout.addWidget(recon_param_oversampling_label)
        recon_param_oversampling_layout.addWidget(recon_param_oversampling_edit)
        self.recon_layout.addLayout(recon_param_oversampling_layout)

        recon_param_iteration_layout = QHBoxLayout()
        recon_param_iteration_label = QLabel(f'Number of Iterations:')
        recon_param_iteration_edit = QLineEdit('100')
        recon_param_iteration_layout.addWidget(recon_param_iteration_label)
        recon_param_iteration_layout.addWidget(recon_param_iteration_edit)
        self.recon_layout.addLayout(recon_param_iteration_layout)

        recon_param_parallel_layout = QHBoxLayout()
        recon_param_parallel_label = QLabel('Parallel Computation:')
        recon_param_parallel_combo = QComboBox()
        recon_param_parallel_combo.addItems(['True', 'False'])
        recon_param_parallel_layout.addWidget(recon_param_parallel_label)
        recon_param_parallel_layout.addWidget(recon_param_parallel_combo)
        self.recon_layout.addLayout(recon_param_parallel_layout)

        recon_output_layout = QHBoxLayout()
        recon_output_label = QLabel(f'Output filename:')
        recon_output_edit = QLineEdit('output_reconstruction')
        recon_output_layout.addWidget(recon_output_label)
        recon_output_layout.addWidget(recon_output_edit)
        self.recon_layout.addLayout(recon_output_layout)

        self.recon_status_label = QLabel("Please click the button below to run reconstruction.")
        self.recon_layout.addWidget(self.recon_status_label)

        self.recon_run_button = QPushButton('Run Reconstruction')
        self.recon_run_button.clicked.connect(lambda:
                                              self.recon_on_click_run(recon_pj_file_edit.text(),
                                                                      recon_angle_file_edit.text(),
                                                                      recon_param_oversampling_edit.text(),
                                                                      recon_param_iteration_edit.text(),
                                                                      recon_param_parallel_combo.currentText(),
                                                                      recon_output_edit.text())
                                              )
        self.recon_layout.addWidget(self.recon_run_button)

        self.recon_thread = None

        self.recon_page.setLayout(self.recon_layout)
        self.stack.addWidget(self.recon_page)
        #####################


        #####################
        # Tracing page
        self.tracing_page = QWidget()
        self.tracing_layout = QVBoxLayout()

        tracing_recon_file_layout = QHBoxLayout()
        tracing_recon_file_label = QLabel('Input Reconstruction File Path:')
        tracing_recon_file_edit = QLineEdit()
        tracing_recon_file_button = QPushButton('Browse')
        tracing_recon_file_button.clicked.connect(lambda _, fe=tracing_recon_file_edit: self.select_file(fe))
        tracing_recon_file_layout.addWidget(tracing_recon_file_label)
        tracing_recon_file_layout.addWidget(tracing_recon_file_edit)
        tracing_recon_file_layout.addWidget(tracing_recon_file_button)
        self.tracing_layout.addLayout(tracing_recon_file_layout)

        tracing_param_max_num_th_layout = QHBoxLayout()
        tracing_param_max_num_th_label = QLabel(f'Max Atom Number Threshold:')
        tracing_param_max_num_th_edit = QLineEdit('100000')
        tracing_param_max_num_th_layout.addWidget(tracing_param_max_num_th_label)
        tracing_param_max_num_th_layout.addWidget(tracing_param_max_num_th_edit)
        self.tracing_layout.addLayout(tracing_param_max_num_th_layout)

        tracing_param_min_atom_dist_layout = QHBoxLayout()
        tracing_param_min_atom_dist_label = QLabel(f'Min Atom Distance (Å):')
        tracing_param_min_atom_dist_edit = QLineEdit('2')
        tracing_param_min_atom_dist_layout.addWidget(tracing_param_min_atom_dist_label)
        tracing_param_min_atom_dist_layout.addWidget(tracing_param_min_atom_dist_edit)
        self.tracing_layout.addLayout(tracing_param_min_atom_dist_layout)

        tracing_output_layout = QHBoxLayout()
        tracing_output_label = QLabel(f'Output filename:')
        tracing_output_edit = QLineEdit('output_tracing')
        tracing_output_layout.addWidget(tracing_output_label)
        tracing_output_layout.addWidget(tracing_output_edit)
        self.tracing_layout.addLayout(tracing_output_layout)

        self.tracing_status_label = QLabel("Please click the button below to run atom tracing.")
        self.tracing_layout.addWidget(self.tracing_status_label)

        self.tracing_run_button = QPushButton('Run Atom Tracing')
        self.tracing_run_button.clicked.connect(lambda:
                                                self.tracing_on_click_run(tracing_recon_file_edit.text(),
                                                                          tracing_param_max_num_th_edit.text(),
                                                                          tracing_param_min_atom_dist_edit.text(),
                                                                          tracing_output_edit.text())
                                                )
        self.tracing_layout.addWidget(self.tracing_run_button)

        self.tracing_thread = None

        self.tracing_page.setLayout(self.tracing_layout)
        self.stack.addWidget(self.tracing_page)
        #####################


        #####################
        # Classification page
        self.class_page = QWidget()
        self.class_layout = QVBoxLayout()

        class_recon_file_layout = QHBoxLayout()
        class_recon_file_label = QLabel('Input Reconstruction File Path:')
        class_recon_file_edit = QLineEdit()
        class_recon_file_button = QPushButton('Browse')
        class_recon_file_button.clicked.connect(lambda _, fe=class_recon_file_edit: self.select_file(fe))
        class_recon_file_layout.addWidget(class_recon_file_label)
        class_recon_file_layout.addWidget(class_recon_file_edit)
        class_recon_file_layout.addWidget(class_recon_file_button)
        self.class_layout.addLayout(class_recon_file_layout)

        class_model_file_layout = QHBoxLayout()
        class_model_file_label = QLabel('Input Model File Path:')
        class_model_file_edit = QLineEdit()
        class_model_file_button = QPushButton('Browse')
        class_model_file_button.clicked.connect(lambda _, fe=class_model_file_edit: self.select_file(fe))
        class_model_file_layout.addWidget(class_model_file_label)
        class_model_file_layout.addWidget(class_model_file_edit)
        class_model_file_layout.addWidget(class_model_file_button)
        self.class_layout.addLayout(class_model_file_layout)

        class_param_species_layout = QHBoxLayout()
        class_param_species_label = QLabel(f'Number of Atom Species:')
        class_param_species_edit = QLineEdit('3')
        class_param_species_layout.addWidget(class_param_species_label)
        class_param_species_layout.addWidget(class_param_species_edit)
        self.class_layout.addLayout(class_param_species_layout)

        class_param_radius_layout = QHBoxLayout()
        class_param_radius_label = QLabel(f'Local Radius (Å):')
        class_param_radius_edit = QLineEdit('10')
        class_param_radius_layout.addWidget(class_param_radius_label)
        class_param_radius_layout.addWidget(class_param_radius_edit)
        self.class_layout.addLayout(class_param_radius_layout)

        class_output_layout = QHBoxLayout()
        class_output_label = QLabel(f'Output filename:')
        class_output_edit = QLineEdit('output_classification')
        class_output_layout.addWidget(class_output_label)
        class_output_layout.addWidget(class_output_edit)
        self.class_layout.addLayout(class_output_layout)

        self.class_status_label = QLabel("Please click the button below to run atom classification.")
        self.class_layout.addWidget(self.class_status_label)

        self.class_run_button = QPushButton('Run Atom Classification')
        self.class_run_button.clicked.connect(lambda:
                                              self.class_on_click_run(class_recon_file_edit.text(),
                                                                      class_model_file_edit.text(),
                                                                      class_param_species_edit.text(),
                                                                      class_param_radius_edit.text(),
                                                                      class_output_edit.text())
                                              )
        self.class_layout.addWidget(self.class_run_button)

        self.class_thread = None

        self.class_page.setLayout(self.class_layout)
        self.stack.addWidget(self.class_page)
        #####################


        #####################
        # Position Refinement page
        self.refine_page = QWidget()
        self.refine_layout = QVBoxLayout()

        refine_pj_file_layout = QHBoxLayout()
        refine_pj_file_label = QLabel('Input Projection File Path:')
        refine_pj_file_edit = QLineEdit()
        refine_pj_file_button = QPushButton('Browse')
        refine_pj_file_button.clicked.connect(lambda _, fe=refine_pj_file_edit: self.select_file(fe))
        refine_pj_file_layout.addWidget(refine_pj_file_label)
        refine_pj_file_layout.addWidget(refine_pj_file_edit)
        refine_pj_file_layout.addWidget(refine_pj_file_button)
        self.refine_layout.addLayout(refine_pj_file_layout)

        refine_angle_file_layout = QHBoxLayout()
        refine_angle_file_label = QLabel('Input Angle File Path:')
        refine_angle_file_edit = QLineEdit()
        refine_angle_file_button = QPushButton('Browse')
        refine_angle_file_button.clicked.connect(lambda _, fe=refine_angle_file_edit: self.select_file(fe))
        refine_angle_file_layout.addWidget(refine_angle_file_label)
        refine_angle_file_layout.addWidget(refine_angle_file_edit)
        refine_angle_file_layout.addWidget(refine_angle_file_button)
        self.refine_layout.addLayout(refine_angle_file_layout)

        refine_model_file_layout = QHBoxLayout()
        refine_model_file_label = QLabel('Input Model File Path:')
        refine_model_file_edit = QLineEdit()
        refine_model_file_button = QPushButton('Browse')
        refine_model_file_button.clicked.connect(lambda _, fe=refine_model_file_edit: self.select_file(fe))
        refine_model_file_layout.addWidget(refine_model_file_label)
        refine_model_file_layout.addWidget(refine_model_file_edit)
        refine_model_file_layout.addWidget(refine_model_file_button)
        self.refine_layout.addLayout(refine_model_file_layout)

        # 添加文件选择功能
        refine_atom_file_layout = QHBoxLayout()
        refine_atom_file_label = QLabel('Input Atom Type File Path:')
        refine_atom_file_edit = QLineEdit()
        refine_atom_file_button = QPushButton('Browse')
        refine_atom_file_button.clicked.connect(lambda _, fe=refine_atom_file_edit: self.select_file(fe))
        refine_atom_file_layout.addWidget(refine_atom_file_label)
        refine_atom_file_layout.addWidget(refine_atom_file_edit)
        refine_atom_file_layout.addWidget(refine_atom_file_button)
        self.refine_layout.addLayout(refine_atom_file_layout)

        refine_param_iteration_layout = QHBoxLayout()
        refine_param_iteration_label = QLabel(f'Number of Iterations:')
        refine_param_iteration_edit = QLineEdit('10')
        refine_param_iteration_layout.addWidget(refine_param_iteration_label)
        refine_param_iteration_layout.addWidget(refine_param_iteration_edit)
        self.refine_layout.addLayout(refine_param_iteration_layout)

        refine_output_layout = QHBoxLayout()
        refine_output_label = QLabel(f'Output filename:')
        refine_output_edit = QLineEdit('output_model')
        refine_output_layout.addWidget(refine_output_label)
        refine_output_layout.addWidget(refine_output_edit)
        self.refine_layout.addLayout(refine_output_layout)

        self.refine_status_label = QLabel("Please click the button below to run position refinement.")
        self.refine_layout.addWidget(self.refine_status_label)

        self.refine_run_button = QPushButton('Run Position Refinement')
        self.refine_run_button.clicked.connect(lambda:
                                               self.refine_on_click_run(refine_pj_file_edit.text(),
                                                                        refine_angle_file_edit.text(),
                                                                        refine_model_file_edit.text(),
                                                                        refine_atom_file_edit.text(),
                                                                        refine_param_iteration_edit.text(),
                                                                        refine_output_edit.text())
                                               )
        self.refine_layout.addWidget(self.refine_run_button)

        self.refine_thread = None

        self.refine_page.setLayout(self.refine_layout)
        self.stack.addWidget(self.refine_page)
        #####################


        # create five buttons for switching pages, as encoded from 0 to 4.
        self.buttons = []
        self.button_layout = QHBoxLayout()
        for i in range(5):
            if i == 0:
                button = QPushButton(f'Home', self)
            if i == 1:
                button = QPushButton(f'Reconstruction', self)
            if i == 2:
                button = QPushButton(f'Atom Tracing', self)
            if i == 3:
                button = QPushButton(f'Classification', self)
            if i == 4:
                button = QPushButton(f'Position Refinement', self)
            button.clicked.connect(lambda _, b=i: self.display_page(b))
            self.buttons.append(button)
            self.button_layout.addWidget(button)


        # window layout.
        self.main_layout = QVBoxLayout()
        self.main_layout.addLayout(self.button_layout)
        self.main_layout.addWidget(self.stack)

        # central widget.
        self.central_widget = QWidget()
        self.central_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.central_widget)

        # Create menu bar
        menubar = self.menuBar()
        analysis_menu = menubar.addMenu('Analysis')

        # Add "PDF Calculator" action
        pdf_calculator_action = QAction('PDF Calculator', self)
        pdf_calculator_action.triggered.connect(self.open_pdf_calculator)
        analysis_menu.addAction(pdf_calculator_action)

        # Add "BOO Calculator" action
        boo_calculator_action = QAction('BOO Calculator', self)
        boo_calculator_action.triggered.connect(self.open_boo_calculator)
        analysis_menu.addAction(boo_calculator_action)

        # Add "CSRO Calculator" action
        csro_calculator_action = QAction('CSRO Calculator', self)
        csro_calculator_action.triggered.connect(self.open_csro_calculator)
        analysis_menu.addAction(csro_calculator_action)

        # Add "Voronoi Calculator" action
        voronoi_calculator_action = QAction('Voronoi Index Calculator', self)
        voronoi_calculator_action.triggered.connect(self.open_voronoi_calculator)
        analysis_menu.addAction(voronoi_calculator_action)

        # initial window size.
        self.setGeometry(200, 200, 500, 500)
        self.setWindowTitle('Atomic Electron Tomography')

    def open_pdf_calculator(self):
        self.pdf_calculator = PDFCalculator()
        self.pdf_calculator.show()

    def open_boo_calculator(self):
        self.boo_calculator = BOOCalculator()
        self.boo_calculator.show()

    def open_csro_calculator(self):
        self.csro_calculator = CSROCalculator()
        self.csro_calculator.show()

    def open_voronoi_calculator(self):
        self.voronoi_calculator = VoronoiCalculator()
        self.voronoi_calculator.show()

    def display_page(self, index):
        self.stack.setCurrentIndex(index)
        # change colors of page buttons.
        for i, button in enumerate(self.buttons):
            if i == index:
                button.setStyleSheet("background-color: lightblue")  # change the color of selected.
            else:
                button.setStyleSheet("")  # other buttons get back to default.

    def select_file(self, line_edit):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "Numpy Files (*.npy);;Matlab Files (*.mat)")
        if file_path:  # 确保用户选择了文件
            line_edit.setText(file_path)  # 更新文本框内容
            print(file_path)

    def recon_on_click_run(self, pj_filename, angle_filename, resire_param_oversampling, resire_param_iteration, resire_param_parallel, output_fn):
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

        param = {}
        param['pj_filename'] = pj_filename
        param['angle_filename'] = angle_filename
        param['resire_param'] = resire_param
        param['output_fn'] = output_fn
        param['job_type'] = 1  # 1:reconstruction, 2: tracing, 3: classification, 4: position refinement

        self.recon_thread = WorkerThread(param)

        self.recon_thread.finished_signal.connect(self.recon_on_thread_finished)
        self.recon_run_button.setDisabled(True)
        self.recon_status_label.setText('Start running reconstruction,  please wait...')
        self.recon_thread.start()

    def recon_on_thread_finished(self, result):
        self.recon_run_button.setDisabled(False)
        self.recon_status_label.setText(result)

    def tracing_on_click_run(self, reconstruction_filename, max_num_th, min_dist, output_fn):
        param = {}
        param['reconstruction_filename'] = reconstruction_filename
        param['max_num_th'] = int(float(max_num_th))
        param['min_dist'] = float(min_dist)
        param['output_fn'] = output_fn
        param['job_type'] = 2  # 1:reconstruction, 2: tracing, 3: classification, 4: position refinement

        self.tracing_thread = WorkerThread(param)

        self.tracing_thread.finished_signal.connect(self.tracing_on_thread_finished)
        self.tracing_run_button.setDisabled(True)
        self.tracing_status_label.setText('Start running atom tracing,  please wait...')
        self.tracing_thread.start()

    def tracing_on_thread_finished(self, result):
        self.tracing_run_button.setDisabled(False)
        self.tracing_status_label.setText(result)

    def class_on_click_run(self, reconstruction_filename, model_filename, num_species, local_radius, output_fn):
        param = {}
        param['reconstruction_filename'] = reconstruction_filename
        param['model_filename'] = model_filename
        param['num_species'] = int(float(num_species))
        param['local_radius'] = float(local_radius)
        param['output_fn'] = output_fn
        param['job_type'] = 3  # 1:reconstruction, 2: tracing, 3: classification, 4: position refinement

        self.class_thread = WorkerThread(param)

        self.class_thread.finished_signal.connect(self.class_on_thread_finished)
        self.class_run_button.setDisabled(True)
        self.class_status_label.setText('Start running atom classification,  please wait...')
        self.class_thread.start()

    def class_on_thread_finished(self, result):
        self.class_run_button.setDisabled(False)
        self.class_status_label.setText(result)

    def refine_on_click_run(self, pj_filename, angle_filename, model_filename, atom_filename, num_iteration, output_fn):
        param = {}
        param['pj_filename'] = pj_filename
        param['angle_filename'] = angle_filename
        param['model_filename'] = model_filename
        param['atom_filename'] = atom_filename
        param['num_iteration'] = int(float(num_iteration))
        param['output_fn'] = output_fn
        param['job_type'] = 4 # 1:reconstruction, 2: tracing, 3: classification, 4: position refinement

        self.refine_thread = WorkerThread(param)

        self.refine_thread.finished_signal.connect(self.refine_on_thread_finished)
        self.refine_run_button.setDisabled(True)
        self.refine_status_label.setText('Start running position refinement,  please wait...')
        self.refine_thread.start()

    def refine_on_thread_finished(self, result):
        self.refine_run_button.setDisabled(False)
        self.refine_status_label.setText(result)

    def run_main_position_refinement(self, projections_file_path, angles_file_path, model_file_path, atoms_file_path, output_fn):
        main_position_refinement(projections_file_path, angles_file_path, model_file_path, atoms_file_path, output_fn)
        return

class WorkerThread(QThread):
    finished_signal = pyqtSignal(object)  # define a signal to show job finished.

    def __init__(self, param):
        super(WorkerThread, self).__init__()
        self.param = param

    def run(self):
        # run the core calculation.
        job_type = self.param['job_type']
        print("job type")
        print(job_type)
        print(type(job_type))
        print(self.param)
        if job_type == 1:  #run reconstruction
            pj_filename = self.param['pj_filename']
            angle_filename = self.param['angle_filename']
            resire_param = self.param['resire_param']
            output_fn = self.param['output_fn']
            main_reconstruction(pj_filename, angle_filename, resire_param, output_fn)
            result = 'Reconstruction done.'
        if job_type == 2:  #run tracing
            reconstruction_filename = self.param['reconstruction_filename']
            max_num_th = self.param['max_num_th']
            min_dist = self.param['min_dist']
            output_fn = self.param['output_fn']
            main_polynomial_tracing(reconstruction_filename, max_num_th, min_dist, output_fn)
            result = 'Tracing done.'
        if job_type == 3:  #run classification
            reconstruction_filename = self.param['reconstruction_filename']
            model_filename = self.param['model_filename']
            num_species = self.param['num_species']
            local_radius = self.param['local_radius']
            output_fn = self.param['output_fn']
            main_classification(reconstruction_filename, model_filename, num_species, local_radius, output_fn)
            result = 'Classification done.'
        if job_type == 4:  #run position refinement
            pj_filename = self.param['pj_filename']
            angle_filename = self.param['angle_filename']
            model_filename = self.param['model_filename']
            atom_filename = self.param['atom_filename']
            num_iteration = self.param['num_iteration']
            output_fn = self.param['output_fn']
            main_position_refinement(pj_filename, angle_filename, model_filename, atom_filename, num_iteration, output_fn)
            result = 'Position refinement done.'
        if job_type == 10:  #run pdf calculator
            model_filename = self.param['model_filename']
            rmax = self.param['rmax']
            output_fn = self.param['output_fn']
            result = calc_pdf(model_filename, rmax=rmax, output_fn=output_fn)
        if job_type == 11:  #run boo calculator
            model_filename = self.param['model_filename']
            cutoff = self.param['cutoff']
            output_fn = self.param['output_fn']
            result = calc_boo(model_filename, cutoff=cutoff, output_fn=output_fn)
        if job_type == 12:  #run csro calculator
            model_filename = self.param['model_filename']
            cutoff = self.param['cutoff']
            output_fn = self.param['output_fn']
            result = calc_csro(model_filename, cutoff=cutoff, output_fn=output_fn)
        if job_type == 13:  #run voronoi calculator
            model_filename = self.param['model_filename']
            # cutoff = self.param['cutoff']
            output_fn = self.param['output_fn']
            result = calc_voronoi(model_filename, output_fn=output_fn)

        self.finished_signal.emit(result)


class PDFCalculator(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('PDF Calculator')
        self.setGeometry(100, 100, 800, 800)

        self.pdf_layout = QVBoxLayout()

        pdf_calculator_model_file_layout = QHBoxLayout()
        pdf_calculator_model_file_label = QLabel('Input Model File Path:')
        pdf_calculator_model_file_edit = QLineEdit()
        pdf_calculator_model_file_button = QPushButton('Browse')
        pdf_calculator_model_file_button.clicked.connect(lambda _, fe=pdf_calculator_model_file_edit: self.select_file(fe))
        pdf_calculator_model_file_layout.addWidget(pdf_calculator_model_file_label)
        pdf_calculator_model_file_layout.addWidget(pdf_calculator_model_file_edit)
        pdf_calculator_model_file_layout.addWidget(pdf_calculator_model_file_button)
        self.pdf_layout.addLayout(pdf_calculator_model_file_layout)

        pdf_calculator_rmax_layout = QHBoxLayout()
        pdf_calculator_rmax_label = QLabel(f'Rmax (Å):')
        pdf_calculator_rmax_edit = QLineEdit('10')
        pdf_calculator_rmax_layout.addWidget(pdf_calculator_rmax_label)
        pdf_calculator_rmax_layout.addWidget(pdf_calculator_rmax_edit)
        self.pdf_layout.addLayout(pdf_calculator_rmax_layout)

        pdf_calculator_output_layout = QHBoxLayout()
        pdf_calculator_output_label = QLabel(f'Output filename:')
        pdf_calculator_output_edit = QLineEdit('output_pdf')
        pdf_calculator_output_layout.addWidget(pdf_calculator_output_label)
        pdf_calculator_output_layout.addWidget(pdf_calculator_output_edit)
        self.pdf_layout.addLayout(pdf_calculator_output_layout)

        self.pdf_status_label = QLabel("Please click the button below to calculate PDF.")
        self.pdf_layout.addWidget(self.pdf_status_label)

        self.pdf_calculator_run_button = QPushButton('Calculate PDF')
        self.pdf_calculator_run_button.clicked.connect(lambda:
                                                        self.plot_pdf_on_click_run(pdf_calculator_model_file_edit.text(),
                                                                                   pdf_calculator_rmax_edit.text(),
                                                                                   pdf_calculator_output_edit.text())
                                                       )
        self.pdf_layout.addWidget(self.pdf_calculator_run_button)

        self.pdf_thread = None

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.pdf_layout.addWidget(self.canvas)

        # Add Matplotlib toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.pdf_layout.addWidget(self.toolbar)

        self.setLayout(self.pdf_layout)

    def select_file(self, line_edit):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "Model Files (*.xyz)")
        if file_path:  # 确保用户选择了文件
            line_edit.setText(file_path)  # 更新文本框内容
            print(file_path)

    def plot_pdf_on_click_run(self, model_filename, rmax, output_fn):
        param = {}
        param['model_filename'] = str(model_filename)
        param['rmax'] = float(rmax)
        param['output_fn'] = output_fn
        param['job_type'] = 10  # 10:pdf, 11:boo, 12:csro, 13:voronoi

        self.pdf_thread = WorkerThread(param)

        self.pdf_thread.finished_signal.connect(self.pdf_on_thread_finished)
        self.pdf_calculator_run_button.setDisabled(True)
        self.pdf_status_label.setText('Start calculating PDF,  please wait...')
        self.pdf_thread.start()

    def pdf_on_thread_finished(self, result):
        self.pdf_calculator_run_button.setDisabled(False)
        self.pdf_status_label.setText("PDF calculation done.")
        x, y = result[0], result[1]
        self.update_plot(x, y)

    def update_plot(self, x, y):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(x, y)
        ax.set_xlabel(r"$r$ ($\mathrm{\AA}$)")
        ax.set_ylabel(r"$g(r)$")
        ax.set_title('RDF')
        self.figure.tight_layout()
        self.canvas.draw()


class BOOCalculator(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('BOO Calculator')
        self.setGeometry(100, 100, 800, 800)

        self.boo_layout = QVBoxLayout()

        boo_calculator_model_file_layout = QHBoxLayout()
        boo_calculator_model_file_label = QLabel('Input Model File Path:')
        boo_calculator_model_file_edit = QLineEdit()
        boo_calculator_model_file_button = QPushButton('Browse')
        boo_calculator_model_file_button.clicked.connect(lambda _, fe=boo_calculator_model_file_edit: self.select_file(fe))
        boo_calculator_model_file_layout.addWidget(boo_calculator_model_file_label)
        boo_calculator_model_file_layout.addWidget(boo_calculator_model_file_edit)
        boo_calculator_model_file_layout.addWidget(boo_calculator_model_file_button)
        self.boo_layout.addLayout(boo_calculator_model_file_layout)

        boo_calculator_rmax_layout = QHBoxLayout()
        boo_calculator_rmax_label = QLabel(f'Cutoff (Å):')
        boo_calculator_rmax_edit = QLineEdit('4.0')
        boo_calculator_rmax_layout.addWidget(boo_calculator_rmax_label)
        boo_calculator_rmax_layout.addWidget(boo_calculator_rmax_edit)
        self.boo_layout.addLayout(boo_calculator_rmax_layout)

        boo_calculator_output_layout = QHBoxLayout()
        boo_calculator_output_label = QLabel(f'Output filename:')
        boo_calculator_output_edit = QLineEdit('output_boo')
        boo_calculator_output_layout.addWidget(boo_calculator_output_label)
        boo_calculator_output_layout.addWidget(boo_calculator_output_edit)
        self.boo_layout.addLayout(boo_calculator_output_layout)

        self.boo_status_label = QLabel("Please click the button below to calculate bond orientational order parameters.")
        self.boo_layout.addWidget(self.boo_status_label)

        self.boo_calculator_run_button = QPushButton('Calculate BOO')
        self.boo_calculator_run_button.clicked.connect(lambda:
                                                        self.plot_boo_on_click_run(boo_calculator_model_file_edit.text(),
                                                                                   boo_calculator_rmax_edit.text(),
                                                                                   boo_calculator_output_edit.text())
                                                       )
        self.boo_layout.addWidget(self.boo_calculator_run_button)

        self.boo_thread = None

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.boo_layout.addWidget(self.canvas)

        # Add Matplotlib toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.boo_layout.addWidget(self.toolbar)

        self.setLayout(self.boo_layout)

    def select_file(self, line_edit):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "Model Files (*.xyz)")
        if file_path:  # 确保用户选择了文件
            line_edit.setText(file_path)  # 更新文本框内容
            print(file_path)

    def plot_boo_on_click_run(self, model_filename, rmax, output_fn):
        param = {}
        param['model_filename'] = str(model_filename)
        param['cutoff'] = float(rmax)
        param['output_fn'] = output_fn
        param['job_type'] = 11  # 10:pdf, 11:boo, 12:csro, 13:voronoi

        self.boo_thread = WorkerThread(param)

        self.boo_thread.finished_signal.connect(self.boo_on_thread_finished)
        self.boo_calculator_run_button.setDisabled(True)
        self.boo_status_label.setText('Start calculating bond orientational order parameters,  please wait...')
        self.boo_thread.start()

    def boo_on_thread_finished(self, result):
        self.boo_calculator_run_button.setDisabled(False)
        self.boo_status_label.setText("BOO calculation done.")
        x, y = result[0], result[1]
        self.update_plot(x, y)

    def update_plot(self, x, y):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)
        # idx = z.argsort()
        # x, y, z = x[idx], y[idx], z[idx]

        sc = ax.scatter(x, y, c=z, s=30, cmap='jet', edgecolors='none')
        ax.set_xlabel("$Q_4$")
        ax.set_ylabel("$Q_6$")
        ax.set_title('BOO Parameters')

        fcc = [0.190941, 0.574524]
        bcc = [0.0363696, 0.510688]
        hcp = [0.09722, 0.484762]
        ax.scatter(fcc[0], fcc[1], c='k', s=50)
        ax.scatter(bcc[0], bcc[1], c='k', s=50)
        ax.scatter(hcp[0], hcp[1], c='k', s=50)
        ax.text(fcc[0] - 0.017, fcc[1]-0.028, 'fcc', fontsize=12)
        ax.text(bcc[0] + 0.007, bcc[1], 'bcc', fontsize=12)
        ax.text(hcp[0] + 0.007, hcp[1], 'hcp', fontsize=12)

        self.figure.colorbar(sc, ax=ax, label='Number of atoms')
        self.figure.tight_layout()
        self.canvas.draw()


class CSROCalculator(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('CSRO Calculator')
        self.setGeometry(100, 100, 800, 800)

        self.csro_layout = QVBoxLayout()

        csro_calculator_model_file_layout = QHBoxLayout()
        csro_calculator_model_file_label = QLabel('Input Model File Path:')
        csro_calculator_model_file_edit = QLineEdit()
        csro_calculator_model_file_button = QPushButton('Browse')
        csro_calculator_model_file_button.clicked.connect(lambda _, fe=csro_calculator_model_file_edit: self.select_file(fe))
        csro_calculator_model_file_layout.addWidget(csro_calculator_model_file_label)
        csro_calculator_model_file_layout.addWidget(csro_calculator_model_file_edit)
        csro_calculator_model_file_layout.addWidget(csro_calculator_model_file_button)
        self.csro_layout.addLayout(csro_calculator_model_file_layout)

        csro_calculator_rmax_layout = QHBoxLayout()
        csro_calculator_rmax_label = QLabel(f'Cutoff (Å):')
        csro_calculator_rmax_edit = QLineEdit('4.0')
        csro_calculator_rmax_layout.addWidget(csro_calculator_rmax_label)
        csro_calculator_rmax_layout.addWidget(csro_calculator_rmax_edit)
        self.csro_layout.addLayout(csro_calculator_rmax_layout)

        csro_calculator_output_layout = QHBoxLayout()
        csro_calculator_output_label = QLabel(f'Output filename:')
        csro_calculator_output_edit = QLineEdit('output_csro')
        csro_calculator_output_layout.addWidget(csro_calculator_output_label)
        csro_calculator_output_layout.addWidget(csro_calculator_output_edit)
        self.csro_layout.addLayout(csro_calculator_output_layout)

        self.csro_status_label = QLabel("Please click the button below to calculate chemical short range order parameters.")
        self.csro_layout.addWidget(self.csro_status_label)

        self.csro_calculator_run_button = QPushButton('Calculate CSRO')
        self.csro_calculator_run_button.clicked.connect(lambda:
                                                        self.plot_csro_on_click_run(csro_calculator_model_file_edit.text(),
                                                                                   csro_calculator_rmax_edit.text(),
                                                                                   csro_calculator_output_edit.text())
                                                       )
        self.csro_layout.addWidget(self.csro_calculator_run_button)

        self.csro_thread = None

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.csro_layout.addWidget(self.canvas)

        # Add Matplotlib toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.csro_layout.addWidget(self.toolbar)

        self.setLayout(self.csro_layout)

    def select_file(self, line_edit):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "Model Files (*.xyz)")
        if file_path:  # 确保用户选择了文件
            line_edit.setText(file_path)  # 更新文本框内容
            print(file_path)

    def plot_csro_on_click_run(self, model_filename, rmax, output_fn):
        param = {}
        param['model_filename'] = str(model_filename)
        param['cutoff'] = float(rmax)
        param['output_fn'] = output_fn
        param['job_type'] = 12  # 10:pdf, 11:boo, 12:csro, 13:voronoi

        self.csro_thread = WorkerThread(param)

        self.csro_thread.finished_signal.connect(self.csro_on_thread_finished)
        self.csro_calculator_run_button.setDisabled(True)
        self.csro_status_label.setText('Start calculating chemical short range order parameters,  please wait...')
        self.csro_thread.start()

    def csro_on_thread_finished(self, result):
        self.csro_calculator_run_button.setDisabled(False)
        self.csro_status_label.setText("CSRO calculation done.")
        x, y = result[0], result[1]
        self.update_plot(x, y)

    def update_plot(self, x, y):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        N = len(x)
        atom_labels = ['Atom{}'.format(i) for i in range(1, N+1)]
        im = ax.imshow(y, cmap = 'RdBu_r', vmin = -1, vmax = 1)

        # Set ticks and labels
        ax.set_xticks(range(len(atom_labels)))
        ax.set_xticklabels(atom_labels)
        ax.set_yticks(range(len(atom_labels)))
        ax.set_yticklabels(atom_labels)

        # add dashed lines.
        ax.set_xticks(np.arange(0.5, N-1, 1), minor=True)
        ax.set_yticks(np.arange(0.5, N-1, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle='--', linewidth=1.0)

        ax.set_xlabel('Atom Type')
        ax.set_ylabel('Atom Type')
        ax.set_title('CSRO Parameters')

        # Add value annotations
        for i in range(len(atom_labels)):
            for j in range(len(atom_labels)):
                ax.text(j, i, f'{y[i, j]:.2f}',
                        ha='center', va='center', fontsize=13, color='black')

        self.figure.colorbar(im, ax=ax, label='Warren-Cowley Parameter')
        self.figure.tight_layout()
        self.canvas.draw()


class VoronoiCalculator(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Voronoi Index Calculator')
        self.setGeometry(100, 100, 800, 800)

        self.voronoi_layout = QVBoxLayout()

        voronoi_calculator_model_file_layout = QHBoxLayout()
        voronoi_calculator_model_file_label = QLabel('Input Model File Path:')
        voronoi_calculator_model_file_edit = QLineEdit()
        voronoi_calculator_model_file_button = QPushButton('Browse')
        voronoi_calculator_model_file_button.clicked.connect(lambda _, fe=voronoi_calculator_model_file_edit: self.select_file(fe))
        voronoi_calculator_model_file_layout.addWidget(voronoi_calculator_model_file_label)
        voronoi_calculator_model_file_layout.addWidget(voronoi_calculator_model_file_edit)
        voronoi_calculator_model_file_layout.addWidget(voronoi_calculator_model_file_button)
        self.voronoi_layout.addLayout(voronoi_calculator_model_file_layout)

        # voronoi_calculator_rmax_layout = QHBoxLayout()
        # voronoi_calculator_rmax_label = QLabel(f'Cutoff (Å):')
        # voronoi_calculator_rmax_edit = QLineEdit('4.0')
        # voronoi_calculator_rmax_layout.addWidget(voronoi_calculator_rmax_label)
        # voronoi_calculator_rmax_layout.addWidget(voronoi_calculator_rmax_edit)
        # self.voronoi_layout.addLayout(voronoi_calculator_rmax_layout)

        voronoi_calculator_output_layout = QHBoxLayout()
        voronoi_calculator_output_label = QLabel(f'Output filename:')
        voronoi_calculator_output_edit = QLineEdit('output_voronoi')
        voronoi_calculator_output_layout.addWidget(voronoi_calculator_output_label)
        voronoi_calculator_output_layout.addWidget(voronoi_calculator_output_edit)
        self.voronoi_layout.addLayout(voronoi_calculator_output_layout)

        self.voronoi_status_label = QLabel("Please click the button below to calculate Voronoi polyhedra indices.")
        self.voronoi_layout.addWidget(self.voronoi_status_label)

        self.voronoi_calculator_run_button = QPushButton('Calculate Voronoi Index')
        self.voronoi_calculator_run_button.clicked.connect(lambda:
                                                        self.plot_voronoi_on_click_run(voronoi_calculator_model_file_edit.text(),
                                                                                   voronoi_calculator_output_edit.text())
                                                       )
        self.voronoi_layout.addWidget(self.voronoi_calculator_run_button)

        self.voronoi_thread = None

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.voronoi_layout.addWidget(self.canvas)

        # Add Matplotlib toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.voronoi_layout.addWidget(self.toolbar)

        self.setLayout(self.voronoi_layout)

    def select_file(self, line_edit):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "Model Files (*.xyz)")
        if file_path:  # 确保用户选择了文件
            line_edit.setText(file_path)  # 更新文本框内容
            print(file_path)

    def plot_voronoi_on_click_run(self, model_filename, output_fn):
        param = {}
        param['model_filename'] = str(model_filename)
        # param['cutoff'] = float(rmax)
        param['output_fn'] = output_fn
        param['job_type'] = 13  # 10:pdf, 11:boo, 12:voronoi, 13:voronoi

        self.voronoi_thread = WorkerThread(param)

        self.voronoi_thread.finished_signal.connect(self.voronoi_on_thread_finished)
        self.voronoi_calculator_run_button.setDisabled(True)
        self.voronoi_status_label.setText('Start calculating Voronoi polyhedra indices,  please wait...')
        self.voronoi_thread.start()

    def voronoi_on_thread_finished(self, result):
        self.voronoi_calculator_run_button.setDisabled(False)
        self.voronoi_status_label.setText("Voronoi index calculation done.")
        x, y = result[0], result[1]
        self.update_plot(x, y)

    def update_plot(self, x, y):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        # only plot top 10.
        ax.bar(x[:10], y[:10], color='tab:blue', edgecolor='black', linewidth=0.8)
        ax.set_xlabel("Voronoi index")
        ax.set_ylabel("Fraction")
        ax.set_title('Top 10 Voronoi indices')
        ax.set_xticklabels(x[:10], rotation=45, ha='right')
        self.figure.tight_layout()
        self.canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    my_app = MyApp()
    my_app.show()
    sys.exit(app.exec_())
