# pyAET

This software rewrites the atomic electron tomography (AET) reconstruction codes in Python.

## Reconstruction layout

Step1 reconstruction is being migrated to two explicit lines:

- `pyaet/resire_numpy/`: stable NumPy/CPP implementation
- `pyaet/resire_torch/`: torch/GPU line

Run instructions for Step1 are in:

- [STEP1_RECONSTRUCTION.md](./STEP1_RECONSTRUCTION.md)


## GUI
For the GUI development, PyQt5 is used.

### Installation
```
pip install PyQt5
pip install PyQt5-tools
```

### Development
1. Run `pyqt5-tools designer` in a command line window to open a QT Designer.

2. Save Your UI File.
Once you're happy with the design, save your project by selecting File > Save. Save it with a .ui extension (e.g., mainwindow.ui). This file contains the XML representation of your GUI layout and properties.

4. Convert UI File to Python Code.
To use your designed GUI with PyQt5 in your Python code, you need to convert the .ui file to a Python file. This can be done using the pyuic5 command-line tool that comes with PyQt5. Open a terminal or command prompt and run:

```
pyuic5 -x yourfile.ui -o yourfile.py
```
This command will generate a yourfile.py containing your GUI code.

4: Use the Generated Code in Your PyQt5 Application.

You can now import and use the generated Python code in your PyQt5 application. Here’s a simple example of how to load and display the main window:

```
import sys
from PyQt5.QtWidgets import QApplication
from yourfile import Ui_MainWindow  # Import the generated class

class MyApplication(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyApplication, self).__init__(parent)
        self.setupUi(self)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyApplication()
    window.show()
    sys.exit(app.exec_())
```
