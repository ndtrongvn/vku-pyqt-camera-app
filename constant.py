from PyQt5.QtWidgets import *

CRITICAL = "critical"
INFO = "info"
WARNING = "warning"
QUESTION = "question"

dialog_type = {
        "critical": "Error!",
        "info": "Info!",
        "warning": "Warning!",
        "question": "Question?"
    }
icon_type = {
    "critical": QMessageBox.Critical,
    "info": QMessageBox.Information,
    "warning": QMessageBox.Warning,
    "question": QMessageBox.Question
}

camera_index = {
    0: "Camera 1",
    1: "Camera 2",
    2: "Camera 3",
    3: "Camera 4"
}