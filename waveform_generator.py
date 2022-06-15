import sys
import numpy as np
import pyqtgraph
import pandas as pd
import scipy.signal as ss
from scipy.fft import fft
from scipy.io.wavfile import write
from PyQt5.QtWidgets import QAction, QApplication, QGridLayout, QHeaderView, QMainWindow, QSpinBox, \
    QDoubleSpinBox, QVBoxLayout, QTableWidget, QWidget, QLabel, QTableWidgetItem, QMessageBox, QFileDialog, \
    QHBoxLayout

__author__ = 'Karol Buchman'
__version__ = 'v.004'


class Generator:
    def __init__(self):
        self.samples = 0
        self.time = 0
        self.t = np.empty(0)
        self.y = np.empty(0)
        self.ff = np.empty(0)
        self.yf = np.empty(0)
        self.f = 0
        self.a = 0
        self.type = -1

    def get_waveform_name(self):
        waveform_names = ('sine wave',
                          'square wave',
                          'sawtooth wave',
                          'triangle wave',
                          'white noise',
                          'none')
        return waveform_names[self.type]

    def set_samples(self, num):
        if num >= 0:
            self.samples = num

    def set_time(self, num):
        if num >= 0:
            self.time = num

    def set_amplitude(self, num):
        self.a = num

    def set_frequency(self, num):
        self.f = num

    def set_type(self, num):
        if num in range(5):
            self.type = num

    def generate_formula(self, t, f, a):
        if self.type == -1:
            return np.nan
        elif self.type == 0:
            return a * np.sin(2 * np.pi * t * f)
        elif self.type == 1:
            return a * ss.square(2 * np.pi * t * f)
        elif self.type == 2:
            return a * ss.sawtooth(2 * np.pi * t * f)
        elif self.type == 3:
            if f != 0:
                p = 1 / f
            else:
                p = 1
            return ((4 * a) / p) * abs((t - (p / 4) % p) % p - (p / 2)) - a
        else:
            return a * (np.random.rand() - 0.5)

    def calculate_y(self):
        if self.time > 0 and self.samples > 0:
            num = 1
            if int(self.samples * self.time) > 1:
                num = int(self.samples * self.time)
            self.t = np.linspace(0, self.time, num)
            y = []
            for i in self.t:
                y.append(self.generate_formula(i, self.f, self.a))
            self.y = np.array(y)

    def calculate_fft(self):
        n = len(self.t)
        if n > 1 and self.y[0] != np.nan:
            dt = self.t[1] - self.t[0]
            self.yf = 2.0 / n * abs(fft(self.y)[0:n // 2])
            self.ff = np.fft.fftfreq(n, d=dt)[0:n // 2]

    def save(self):
        file_filter_options = 'Time sequence - text file *.csv;;' \
                              'Fourier transform - text file *.csv;;' \
                              'Waveform sound - audio file *.wav'
        file_name, file_filter = QFileDialog.getSaveFileName(caption='Save data file',
                                                             filter=file_filter_options)

        if file_name != '':
            if file_filter == 'Time sequence - text file *.csv':
                df = pd.DataFrame({'time': self.t, 'amplitude': self.y})
                df.to_csv(file_name, index=False, sep='\t')
            elif file_filter == 'Fourier transform - text file *.csv':
                df = pd.DataFrame({'frequency': self.ff, 'amplitude': self.yf})
                df.to_csv(file_name, index=False, sep='\t')
            elif file_filter == 'Waveform sound - audio file *.wav':
                audio_data = np.int16(self.y * 2 ** 15)
                write(file_name, self.samples, audio_data)

    @staticmethod
    def exit():
        sys.exit()


class GeneratorGUI(QMainWindow):
    def __init__(self, gen):
        super().__init__()
        QApplication.processEvents()
        self.gen = gen

        # main window options
        self.title = 'Waveform generator'
        self.height = 500
        self.width = 500
        self.left = 700
        self.top = 500
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # creating whole GUI
        self._create_layouts()
        self._create_menu_bar()
        self._create_header()
        self.table_rows_count = 0
        self._create_table()
        self.spinbox_values = [0, 0, 0, 0]
        self._create_spinbox()
        self._create_double_spinboxes()
        self._create_plots()
        self.refresh()

    def _create_layouts(self):
        self.main_layout = QHBoxLayout()
        self.left_layout = QVBoxLayout()
        self.right_layout = QVBoxLayout()

        self.main_layout.addLayout(self.left_layout, 1)
        self.main_layout.addLayout(self.right_layout, 3)

        self._main_widget = QWidget(self)
        self.setCentralWidget(self._main_widget)
        self._main_widget.setLayout(self.main_layout)

        self.header_layout = QVBoxLayout()
        self.spinboxes_layout = QGridLayout()
        self.table_layout = QGridLayout()
        self.plots_layout = QVBoxLayout()

        self.left_layout.addLayout(self.header_layout)
        self.left_layout.addLayout(self.table_layout)
        self.left_layout.addLayout(self.spinboxes_layout)

        self.right_layout.addLayout(self.plots_layout)

    def _create_header(self):
        self.current_waveform_name = QLabel(f'Current waveform type: {self.gen.get_waveform_name()}', self)
        self.header_layout.addWidget(self.current_waveform_name)

    def _create_menu_bar(self):
        self.menu_bar = self.menuBar()

        file_ = self.menu_bar.addMenu('File')
        calc_ = self.menu_bar.addMenu('Calculate')
        help_ = self.menu_bar.addMenu('Help')

        self.save_time_waveform_ = QAction('Save', self)
        self.save_time_waveform_.triggered.connect(self.gen.save)

        self.exit_ = QAction('Exit', self)
        self.exit_.triggered.connect(self.gen.exit)

        self.about_ = QAction('About', self)
        self.about_.triggered.connect(self.about)

        self.sine_ = QAction('Sine wave', self)
        self.sine_.triggered.connect(lambda: self.set_type(0))

        self.square_ = QAction('Square wave', self)
        self.square_.triggered.connect(lambda: self.set_type(1))

        self.sawtooth_ = QAction('Sawtooth wave', self)
        self.sawtooth_.triggered.connect(lambda: self.set_type(2))

        self.triangle_ = QAction('Triangle wave', self)
        self.triangle_.triggered.connect(lambda: self.set_type(3))

        self.noise_ = QAction('White noise', self)
        self.noise_.triggered.connect(lambda: self.set_type(4))

        file_.addAction(self.save_time_waveform_)
        file_.addAction(self.exit_)
        help_.addAction(self.about_)
        calc_.addAction(self.sine_)
        calc_.addAction(self.sawtooth_)
        calc_.addAction(self.square_)
        calc_.addAction(self.triangle_)
        calc_.addAction(self.noise_)

    def _create_table(self):
        self.table = QTableWidget()
        self.table.setRowCount(self.table_rows_count)
        self.table.setColumnCount(2)

        # setting max size of columns' available space
        header_ = self.table.horizontalHeader()
        header_.setSectionResizeMode(QHeaderView.ResizeToContents)
        header_.setSectionResizeMode(0, QHeaderView.Stretch)
        header_.setSectionResizeMode(1, QHeaderView.Stretch)

        # setting names for each of columns
        self.table.setHorizontalHeaderLabels(['time', 'amplitude'])
        self.table_layout.addWidget(self.table)

    def set_table_rows_count(self, num):
        self.table.setRowCount(num)

    def _create_spinbox(self):
        self._spinboxes = []
        self.spinbox = QSpinBox()
        self.spinbox.setValue(0)
        self.spinbox.setMinimum(0)
        self.spinbox.setMaximum(1000000)
        self.spinboxes_layout.addWidget(self.spinbox, 1, 0)
        self.spinboxes_layout.addWidget(QLabel('Samples per second: '), 0, 0)
        self.spinbox.valueChanged.connect(lambda: self.gen.set_samples(self.spinbox.value()))
        self._spinboxes.append(self.spinbox)

    def _create_double_spinboxes(self):
        self.d_spinbox0 = QDoubleSpinBox()
        self.d_spinbox0.setSingleStep(0.01)
        self.d_spinbox0.setValue(0)
        self.d_spinbox0.setMinimum(0.00)
        self.d_spinbox0.setMaximum(1000000)
        self.spinboxes_layout.addWidget(self.d_spinbox0, 4, 0)
        self.spinboxes_layout.addWidget(QLabel('Amplitude: '), 3, 0)
        self.d_spinbox0.valueChanged.connect(lambda: self.gen.set_amplitude(self.d_spinbox0.value()))
        self._spinboxes.append(self.d_spinbox0)

        self.d_spinbox1 = QDoubleSpinBox()
        self.d_spinbox1.setSingleStep(0.01)
        self.d_spinbox1.setValue(0)
        self.d_spinbox1.setMinimum(0.00)
        self.d_spinbox1.setMaximum(1000000)
        self.spinboxes_layout.addWidget(self.d_spinbox1, 1, 1)
        self.spinboxes_layout.addWidget(QLabel('Time: '), 0, 1)
        self.d_spinbox1.valueChanged.connect(lambda: self.gen.set_time(self.d_spinbox1.value()))
        self._spinboxes.append(self.d_spinbox1)

        self.d_spinbox2 = QDoubleSpinBox()
        self.d_spinbox2.setSingleStep(0.01)
        self.d_spinbox2.setValue(0)
        self.d_spinbox2.setMaximum(1000000)
        self.spinboxes_layout.addWidget(self.d_spinbox2, 4, 1)
        self.spinboxes_layout.addWidget(QLabel('Frequency: '), 3, 1)
        self.d_spinbox2.valueChanged.connect(lambda: self.gen.set_frequency(self.d_spinbox2.value()))
        self._spinboxes.append(self.d_spinbox2)

    def _create_plots(self):
        self.plot1_header = QLabel('', self)
        self.plots_layout.addWidget(self.plot1_header)

        self.waveform_graph = pyqtgraph.PlotWidget()
        self.plots_layout.addWidget(self.waveform_graph)
        pen1 = pyqtgraph.mkPen(color=(255, 128, 255), width=2)
        self.waveform_plot = self.waveform_graph.plot(self.gen.t, self.gen.y, pen=pen1)
        self.waveform_graph.setTitle('Time sequence')
        self.waveform_graph.setLabel('left', text='<font size=4>amplitude</font>')
        self.waveform_graph.setLabel('bottom', text='<font size=4>time [s]</font>')

        self.plot2_header = QLabel('', self)
        self.plots_layout.addWidget(self.plot2_header)

        self.fft_graph = pyqtgraph.PlotWidget()
        self.plots_layout.addWidget(self.fft_graph)
        pen2 = pyqtgraph.mkPen(color=(255, 128, 0), width=2)
        self.fft_plot = self.fft_graph.plot(self.gen.ff, self.gen.yf, pen=pen2)
        self.fft_graph.setTitle('Fourier transformation')
        self.fft_graph.setLabel('left', text='<font size=4>amplitude</font>')
        self.fft_graph.setLabel('bottom', text='<font size=4>frequency [Hz]</font>')

    def update_table(self):
        t = len(self.gen.t)
        if t > 0:
            self.set_table_rows_count(t)
            for i in range(t):
                self.table.setItem(i, 0, QTableWidgetItem(str(self.gen.t[i])))
                self.table.setItem(i, 1, QTableWidgetItem(str(self.gen.y[i])))

    def update_plots(self):
        self.waveform_plot.setData(self.gen.t, self.gen.y)
        self.fft_plot.setData(self.gen.ff, self.gen.yf)

    def update_all(self):
        self.gen.calculate_y()
        self.gen.calculate_fft()
        self.current_waveform_name.setText(f'Current waveform type: {self.gen.get_waveform_name()}')
        self.update_plots()
        self.update_table()

    def refresh(self):
        for i in self._spinboxes:
            i.valueChanged.connect(self.update_all)

    def set_type(self, num):
        self.gen.set_type(num)
        self.update_all()

    @staticmethod
    def about():
        ab = QMessageBox()
        ab.setIcon(QMessageBox.Information)
        ab.setText('Program for creating waveforms.')
        ab.setWindowTitle('About')
        ab.setStandardButtons(QMessageBox.Ok)
        ab.exec()


def main():
    generator = QApplication(sys.argv)
    cal = Generator()
    gui = GeneratorGUI(cal)
    gui.showMaximized()
    sys.exit(generator.exec_())


while True:
    main()
