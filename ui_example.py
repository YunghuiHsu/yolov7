import sys
import cv2
from PyQt5.QtCore import Qt, QThread
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QCheckBox
from Camera import Camera_YOLOv7

        
class DemoButton(QPushButton):
    def __init__(self, parent):
        super().__init__('Select Demo video', parent)
        self.clicked.connect(self.start_demo)
        
        self.use_webcam = False
        self.webcam_button = QPushButton('Use Webcam', parent)
        self.webcam_button.setCheckable(True)
        self.webcam_button.setChecked(False)
        self.webcam_button.toggled.connect(self.toggle_webcam)

    def start_demo(self):
        video_path, _ = QFileDialog.getOpenFileName(self, 'Select demo video')
        if video_path:
            self.parent().parent().start_video_thread(video_path)
    def toggle_webcam(self, checked):
        self.use_webcam = checked
        if checked:
            self.parent().parent().start_video_thread(str(0))  # 0 means using the default webcam



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.video_thread = None
        self.use_webcam = False

    def init_ui(self):
        self.setWindowTitle('Pose Detection Demo')
        self.setGeometry(100, 100, 800, 600)

        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(self.video_label)

        toolbar = self.addToolBar("Toolbar")
        demo_button = DemoButton(self)
        toolbar.addWidget(demo_button)
        toolbar.addWidget(demo_button.webcam_button)
        self.show()

    def toggle_webcam(self, checked):
        self.use_webcam = checked
    
    def update_video_label(self, qimage):
        pixmap = QPixmap.fromImage(qimage)
        self.video_label.setPixmap(pixmap)
        

    
    def start_video_thread(self, video_path):
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.terminate()
        
        print(f'use_webcam : {self.use_webcam}')
        if self.use_webcam:
            print("WebCamera Launch")
            self.video_thread = Camera_YOLOv7(str(0), 640)  # 0 means using the default webcam
        else:
            print("Demo Video Launch")
            self.video_thread = Camera_YOLOv7(video_path, 640)
            
        self.video_thread.change_pixmap_signal.connect(self.update_video_label)
        self.video_thread.start()
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = MainWindow()
    sys.exit(app.exec_())