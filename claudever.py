import sys
import qtawesome as qta
import json
import os
import threading
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
import whisper
from PyQt5.QtCore import QThread, pyqtSignal, QRunnable, QThreadPool, QObject
from datetime import datetime, timedelta
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtCore import QTimer
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import shutil
import traceback

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/calendar']

# --- Whisper Integration: New Implementation ---
class WhisperWorker(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    status = pyqtSignal(str)

    def __init__(self, duration=5, parent=None):
        super().__init__(parent)
        self.duration = duration
        self.sample_rate = 16000
        self.temp_file = os.path.join(os.path.expanduser("~"), ".seinxcal_temp_new.wav")
        self.language = "en"

    def set_language(self, lang):
        self.language = lang

    def run(self):
        import logging
        import torch
        try:
            logging.info(f"[WhisperWorker] Checking ffmpeg: {shutil.which('ffmpeg')}")
            if shutil.which("ffmpeg") is None:
                self.error.emit("ffmpeg is not installed or not in your PATH. Please install ffmpeg and try again.")
                logging.error("ffmpeg not found in PATH.")
                return
            self.status.emit("Recording audio...")
            try:
                logging.info(f"[WhisperWorker] Recording to temp file: {self.temp_file}")
                recording = sd.rec(int(self.duration * self.sample_rate), samplerate=self.sample_rate, channels=1, dtype='int16')
                sd.wait()
                wavfile.write(self.temp_file, self.sample_rate, recording)
                logging.info(f"[WhisperWorker] Temp file exists after recording: {os.path.exists(self.temp_file)}")
            except Exception as e:
                logging.error(f"[WhisperWorker] Audio recording failed: {e}\n{traceback.format_exc()}")
                self.error.emit(f"Audio recording failed: {e}")
                return
            self.status.emit("Transcribing...")
            try:
                logging.info(f"[WhisperWorker] Checking temp file before transcription: {self.temp_file}, exists: {os.path.exists(self.temp_file)}")
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                logging.info(f"[WhisperWorker] Using device: {device}")
                model = whisper.load_model("base", device=device)
                result = model.transcribe(self.temp_file, language=self.language)
                text = result.get("text", "").strip()
                if not text:
                    self.error.emit("No speech detected. Please try again.")
                else:
                    self.finished.emit(text)
            except Exception as e:
                logging.error(f"[WhisperWorker] Transcription failed: {e}\n{traceback.format_exc()}")
                self.error.emit(f"pTranscription failed: {e}")
            finally:
                if os.path.exists(self.temp_file):
                    os.remove(self.temp_file)
        except Exception as e:
            logging.error(f"[WhisperWorker] Unexpected error: {e}\n{traceback.format_exc()}")
            self.error.emit(str(e))

class SpeechToTextWidget(QWidget):
    textCaptured = pyqtSignal(str)
    def __init__(self, parent=None, target_field=None):
        super().__init__(parent)
        self.target_field = target_field
        layout = QHBoxLayout()
        self.mic_button = QPushButton()
        self.mic_button.setIcon(qta.icon('fa5s.microphone'))
        self.mic_button.setToolTip("Click to use voice input for this field")
        self.mic_button.clicked.connect(self.start_listening)
        layout.addWidget(self.mic_button)
        self.setLayout(layout)
        self.worker = None
        self.language = "en"
    def set_language(self, lang):
        self.language = lang
        if self.worker is not None:
            self.worker.set_language(lang)
    def start_listening(self):
        self.mic_button.setEnabled(False)
        self.worker = WhisperWorker()
        self.worker.set_language(self.language)
        self.worker.finished.connect(self.on_transcription_complete)
        self.worker.error.connect(self.on_transcription_error)
        self.worker.status.connect(self.on_status_update)
        self.worker.start()
    def on_transcription_complete(self, text):
        self.mic_button.setEnabled(True)
        self.textCaptured.emit(text)
        if self.target_field is not None:
            if isinstance(self.target_field, QTextEdit):
                self.target_field.append(text)
            else:
                self.target_field.setText(text)
    def on_transcription_error(self, error_msg):
        self.mic_button.setEnabled(True)
        QMessageBox.warning(self, "Speech Recognition Error", error_msg)
    def on_status_update(self, status):
        self.mic_button.setToolTip(status)


class ListeningOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Make it a popup that stays on top
        self.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Create semi-transparent dark background
        self.setStyleSheet("""
            ListeningOverlay {
                background-color: rgba(0, 0, 0, 0.7);
                border-radius: 10px;
            }
            QLabel {
                color: white;
                background-color: transparent;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Add microphone icon
        self.mic_label = QLabel()
        self.update_mic_icon()
        layout.addWidget(self.mic_label, alignment=Qt.AlignCenter)
        
        # Status label
        self.status_label = QLabel("Initializing...")
        self.status_label.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
        """)
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Progress indicator
        self.progress_label = QLabel()
        self.progress_label.setStyleSheet("font-size: 14px;")
        self.progress_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.progress_label)
        
        self.setLayout(layout)
        
        # Animation
        self.dots = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(500)
        
        # Set fixed size for the overlay
        self.setFixedSize(250, 150)
        
        self.current_status = ""
    
    def update_mic_icon(self, recording=False):
        icon = qta.icon('fa5s.microphone' + ('-slash' if not recording else ''), 
                       color='#4CAF50' if recording else 'white')
        self.mic_label.setPixmap(icon.pixmap(32, 32))
    
    def update_status(self, status):
        self.current_status = status
        self.status_label.setText(status)
        
        # Update microphone icon based on status
        if status == "Recording audio...":
            self.update_mic_icon(True)
        else:
            self.update_mic_icon(False)
    
    def update_animation(self):
        self.dots = (self.dots + 1) % 4
        self.progress_label.setText("." * self.dots)
    
    def showEvent(self, event):
        if self.parent():
            # Center the overlay on the parent widget
            parent_rect = self.parent().rect()
            parent_center = self.parent().mapToGlobal(parent_rect.center())
            self.move(parent_center.x() - self.width() // 2,
                     parent_center.y() - self.height() // 2)
        super().showEvent(event)

class LoginDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Login")
        self.setFixedSize(400, 150)
        self.calendar_id = None
        self.credentials = None
        self.user_email = None
        
        layout = QVBoxLayout()
        
        # Calendar ID input
        form_layout = QFormLayout()
        self.calendar_id_input = QLineEdit()
        self.calendar_id_input.setPlaceholderText("Enter your calendar ID")
        form_layout.addRow("Calendar ID:", self.calendar_id_input)
        layout.addLayout(form_layout)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.login)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.setLayout(layout)
        
    def login(self):
        self.calendar_id = self.calendar_id_input.text().strip()
        if not self.calendar_id:
            QMessageBox.warning(self, "Error", "Please enter a calendar ID")
            return
            
        try:
            creds = None
            if os.path.exists('token.json'):
                creds = Credentials.from_authorized_user_file('token.json', SCOPES)
            
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        'credentials.json', SCOPES)
                    creds = flow.run_local_server(port=0)
                
                with open('token.json', 'w') as token:
                    token.write(creds.to_json())
            
            # Test the connection with provided calendar ID
            service = build('calendar', 'v3', credentials=creds)
            calendar = service.calendars().get(calendarId=self.calendar_id).execute()
            
            self.user_email = calendar.get('id', 'Unknown')
            self.credentials = creds
            self.accept()
            
        except Exception as e:
            QMessageBox.warning(self, "Authentication Error", f"Failed to authenticate: {str(e)}")
            return

class DateSearchDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Search Calendar by Date")
        self.setFixedSize(300, 150)
        
        layout = QVBoxLayout()
        
        label = QLabel("Select a date to search:")
        layout.addWidget(label)
        
        self.date_edit = QDateEdit()
        self.date_edit.setDate(QDate.currentDate())
        self.date_edit.setCalendarPopup(True)
        layout.addWidget(self.date_edit)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.setLayout(layout)
    
    def get_date(self):
        return self.date_edit.date().toPyDate()

class AddEventDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Event")
        self.setFixedSize(400, 320)
        
        layout = QVBoxLayout()
        
        # Event name with speech input
        name_container = QWidget()
        name_layout = QHBoxLayout(name_container)
        name_layout.setContentsMargins(0, 0, 0, 0)
        name_label = QLabel("Event Name:")
        self.name_edit = QLineEdit()
        self.name_speech = SpeechToTextWidget(target_field=self.name_edit)
        self.name_speech.textCaptured.connect(lambda text: self.name_edit.setText(text))
        name_layout.addWidget(name_label)
        name_layout.addWidget(self.name_edit)
        name_layout.addWidget(self.name_speech)
        layout.addWidget(name_container)
        
        # Location with speech input
        location_container = QWidget()
        location_layout = QHBoxLayout(location_container)
        location_layout.setContentsMargins(0, 0, 0, 0)
        location_label = QLabel("Location:")
        self.location_edit = QLineEdit()
        self.location_speech = SpeechToTextWidget(target_field=self.location_edit)
        self.location_speech.textCaptured.connect(lambda text: self.location_edit.setText(text))
        location_layout.addWidget(location_label)
        location_layout.addWidget(self.location_edit)
        location_layout.addWidget(self.location_speech)
        layout.addWidget(location_container)
        
        # All Day checkbox
        self.all_day_check = QCheckBox("All Day Event")
        self.all_day_check.stateChanged.connect(self.on_all_day_changed)
        layout.addWidget(self.all_day_check)
        
        # Start date/time
        layout.addWidget(QLabel("Start Date & Time:"))
        self.start_datetime = QDateTimeEdit()
        self.start_datetime.setDateTime(QDateTime.currentDateTime())
        self.start_datetime.setCalendarPopup(True)
        self.start_datetime.setDisplayFormat("yyyy-MM-dd HH:mm")
        layout.addWidget(self.start_datetime)
        
        # End date/time
        layout.addWidget(QLabel("End Date & Time:"))
        self.end_datetime = QDateTimeEdit()
        self.end_datetime.setDateTime(QDateTime.currentDateTime().addSecs(3600))
        self.end_datetime.setCalendarPopup(True)
        layout.addWidget(self.end_datetime)
        
        # Remarks with speech input
        remarks_container = QWidget()
        remarks_layout = QHBoxLayout(remarks_container)
        remarks_layout.setContentsMargins(0, 0, 0, 0)
        remarks_label = QLabel("Remarks:")
        self.remarks_edit = QTextEdit()
        self.remarks_edit.setMaximumHeight(60)
        self.remarks_speech = SpeechToTextWidget(target_field=self.remarks_edit)
        self.remarks_speech.textCaptured.connect(lambda text: self.remarks_edit.append(text))
        remarks_layout.addWidget(remarks_label)
        remarks_layout.addWidget(self.remarks_edit)
        remarks_layout.addWidget(self.remarks_speech, alignment=Qt.AlignTop)
        layout.addWidget(remarks_container)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.setLayout(layout)
    
    def setup_datetime_section(self, date_edit, label, show_time=True):
        # Create a horizontal layout for the date/time section
        section_layout = QHBoxLayout()
        section_layout.addWidget(QLabel(label))
        
        # Add date edit
        self.date_part = QDateEdit(date_edit)
        self.date_part.setCalendarPopup(True)
        section_layout.addWidget(self.date_part)
        
        # Add time edit
        if show_time:
            self.time_part = QTimeEdit(date_edit)
            section_layout.addWidget(self.time_part)
        
        return section_layout

    def on_all_day_changed(self, state):
        # Store current times before changing format
        start_time = self.start_datetime.time()
        end_time = self.end_datetime.time()
        
        # Change display format based on all-day status
        self.start_datetime.setDisplayFormat("yyyy-MM-dd" if state else "yyyy-MM-dd HH:mm")
        self.end_datetime.setDisplayFormat("yyyy-MM-dd" if state else "yyyy-MM-dd HH:mm")
        
        if state:
            # For all-day events, set times to start and end of day
            self.start_datetime.setTime(QTime(0, 0))
            self.end_datetime.setTime(QTime(23, 59))
        else:
            # Restore previous times when switching back to non-all-day
            self.start_datetime.setTime(start_time)
            self.end_datetime.setTime(end_time)
    
    def get_event_data(self):
        is_all_day = self.all_day_check.isChecked()
        start_dt = self.start_datetime.dateTime().toPyDateTime()
        end_dt = self.end_datetime.dateTime().toPyDateTime()
        
        return {
            'name': self.name_edit.text(),
            'location': self.location_edit.text(),
            'start': start_dt.date() if is_all_day else start_dt,
            'end': end_dt.date() if is_all_day else end_dt,
            'remarks': self.remarks_edit.toPlainText(),
            'is_all_day': is_all_day
        }

class CalendarTable(QTableWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_app = parent
        self.setColumnCount(5)
        self.setHorizontalHeaderLabels(['Name', 'Location', 'Start Date', 'End Date', 'Remarks'])
        self.event_data = {}  # Store event data by row
        
        # Make table responsive
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Add resize event handler to maintain proportions
        self.viewport().installEventFilter(self)
        
        # Set relative column widths with proportional sizes
        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Interactive)  # Name
        header.setSectionResizeMode(1, QHeaderView.Interactive)  # Location
        header.setSectionResizeMode(2, QHeaderView.Interactive)  # Start Date
        header.setSectionResizeMode(3, QHeaderView.Interactive)  # End Date
        header.setSectionResizeMode(4, QHeaderView.Stretch)     # Remarks
        
        # Set proportional widths (total should be less than viewport width)
        total_width = self.viewport().width()
        self.setColumnWidth(0, int(total_width * 0.25))  # Name - 22%
        self.setColumnWidth(1, int(total_width * 0.25))  # Location - 15%
        self.setColumnWidth(2, int(total_width * 0.18))  # Start Date - 18%
        self.setColumnWidth(3, int(total_width * 0.18))  # End Date - 18%
        # Remarks gets remaining ~27% automatically due to Stretch mode
        
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setAlternatingRowColors(True)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)  # Disable editing
        self.cellClicked.connect(self.handle_event_cell_click)
        
        #empty rows for adding new events
        self.setRowCount(50)
        
        # Create actions widget
        self.actions_widget = None
    
    def handle_event_cell_click(self, row, column):
        # Only show actions for event rows (not empty, not separator)
        item = self.item(row, 0)
        if item is None or item.text() == "":
            self.parent_app.add_event()
            return
        # Don't show for separator row
        if item.text() == "Upcoming Events":
            return
        if self.actions_widget:
            self.actions_widget.hide()
            self.actions_widget.deleteLater()
        self.show_actions_widget(row)
        # Enable mouse tracking to detect when cursor leaves the row
        self.setMouseTracking(True)

    def show_actions_widget(self, row):
        event_data = self.event_data.get(row)
        if not event_data:
            return
        self.actions_widget = QWidget(self)
        layout = QHBoxLayout(self.actions_widget)
        layout.setSpacing(3)
        layout.setContentsMargins(0, 0, 0, 0)

        # Edit button with icon
        edit_btn = QPushButton(self.actions_widget)
        edit_icon = QIcon.fromTheme("edit", QIcon("icons/edit.png"))
        edit_btn.setIcon(edit_icon)
        edit_btn.setToolTip("Edit")
        edit_btn.setStyleSheet("border: none; background: transparent;")
        edit_btn.setCursor(Qt.PointingHandCursor)
        edit_btn.clicked.connect(lambda: self.parent_app.update_event(event_data))

        # Delete button with icon
        delete_btn = QPushButton(self.actions_widget)
        delete_icon = QIcon.fromTheme("delete", QIcon("icons/delete.png"))
        delete_btn.setIcon(delete_icon)
        delete_btn.setToolTip("Delete")
        delete_btn.setStyleSheet("border: none; background: transparent;")
        delete_btn.setCursor(Qt.PointingHandCursor)
        delete_btn.clicked.connect(lambda: self.parent_app.delete_event(event_data))

        layout.addWidget(edit_btn)
        layout.addWidget(delete_btn)

        # Position the buttons container in the remarks column
        rect = self.visualItemRect(self.item(row, 4))
        self.actions_widget.setFixedSize(60, rect.height()-2)
        horizontal_pos = rect.x() + rect.width() - 65
        vertical_pos = rect.y() - 1
        if horizontal_pos < rect.x():
            horizontal_pos = rect.x() + 5
        self.actions_widget.move(horizontal_pos, vertical_pos)
        self.actions_widget.show()
    
    def show_actions_menu(self, row, event_data):
        menu = QMenu(self)
        update_action = menu.addAction("Update")
        delete_action = menu.addAction("Delete")
        
        update_action.triggered.connect(lambda: self.parent_app.update_event(event_data))
        delete_action.triggered.connect(lambda: self.parent_app.delete_event(event_data))
        
        # Show menu at mouse position
        menu.exec_(QCursor.pos())
    
    def leaveEvent(self, event):
        # When mouse leaves the table completely, disable tracking and hide actions
        self.setMouseTracking(False)
        if self.actions_widget:
            if not self.underMouse():
                self.actions_widget.hide()
                self.actions_widget.deleteLater()
                self.actions_widget = None
        super().leaveEvent(event)
        
    def mouseMoveEvent(self, event):
        # Check if mouse is still in the same row as the action buttons
        current_row = self.rowAt(event.y())
        if self.actions_widget and current_row != self.currentRow():
            self.actions_widget.hide()
            self.actions_widget.deleteLater()
            self.actions_widget = None
            self.setMouseTracking(False)
        super().mouseMoveEvent(event)
        
    def eventFilter(self, obj, event):
        # Handle resize events to maintain column proportions
        if obj == self.viewport() and event.type() == QEvent.Resize:
            width = self.viewport().width()
            self.setColumnWidth(0, int(width * 0.22))  # Name
            self.setColumnWidth(1, int(width * 0.15))  # Location
            self.setColumnWidth(2, int(width * 0.18))  # Start Date
            self.setColumnWidth(3, int(width * 0.18))  # End Date
            # Remarks column (index 4) will adjust automatically due to Stretch mode
        return super().eventFilter(obj, event)

class UpdateEventDialog(AddEventDialog):
    def __init__(self, event_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Update Event")
        
        # Pre-fill the fields with existing event data
        self.name_edit.setText(event_data.get('summary', ''))
        self.location_edit.setText(event_data.get('location', ''))
        self.remarks_edit.setText(event_data.get('description', ''))
        
        # First determine if this is an all-day event and set the checkbox
        # An event is all-day if it uses 'date' instead of 'dateTime'
        is_all_day = 'date' in event_data['start']
        self.all_day_check.setChecked(is_all_day)
        
        # Update the datetime display format based on all-day status
        if is_all_day:
            self.start_datetime.setDisplayFormat("yyyy-MM-dd")
            self.end_datetime.setDisplayFormat("yyyy-MM-dd")
        else:
            self.start_datetime.setDisplayFormat("yyyy-MM-dd HH:mm")
        
        # Parse start and end times
        start = event_data['start'].get('dateTime', event_data['start'].get('date'))
        end = event_data['end'].get('dateTime', event_data['end'].get('date'))
        
        # Handle both datetime and date-only formats
        if 'T' in start:
            # Has time component
            start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
        else:
            # Date only - use start of day
            start_dt = datetime.fromisoformat(start)
        
        if 'T' in end:
            # Has time component
            end_dt = datetime.fromisoformat(end.replace('Z', '+00:00'))
        else:
            # Date only - use end of day
            end_dt = datetime.fromisoformat(end)
        
        # Convert to local time and set in the dialog
        self.start_datetime.setDateTime(QDateTime(
            QDate(start_dt.year, start_dt.month, start_dt.day),
            QTime(start_dt.hour, start_dt.minute)
        ))
        
        self.end_datetime.setDateTime(QDateTime(
            QDate(end_dt.year, end_dt.month, end_dt.day),
            QTime(end_dt.hour, end_dt.minute)
        ))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.service = None
        self.user_email = ""
        self.calendar_id = None
        self.current_date = datetime.now().date()
        self.theme = "light"
        self.language = "en"
        
        # Set minimum size and get screen geometry
        self.setMinimumSize(1000, 600)
        screen = QDesktopWidget().availableGeometry()
        width = int(screen.width() * 0.8)  # 80% of screen width
        height = int(screen.height() * 0.8)  # 80% of screen height
        
        # Set size and center the window
        self.resize(width, height)
        self.move((screen.width() - width) // 2, (screen.height() - height) // 2)
        
        self.setWindowIcon(QIcon('calendar-app-50.png'))
        self.setWindowTitle("SEINXCAL")
        
        # Setup auto-refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.load_events)
        self.refresh_timer.setInterval(30000)  # 30 seconds
        
        self.setup_ui()
        self.apply_theme()
        self.user_label.setText("No connected account")
    
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(10)  # Add spacing between widgets
        layout.setContentsMargins(10, 10, 10, 10)  # Add margins
        
        # User info and date
        info_bar = self.create_info_bar()
        layout.addWidget(info_bar)
        
        # Tab widget that expands to fill space
        tab_bar = self.create_tab_bar()
        tab_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(tab_bar, 1)  # The 1 gives it a stretch factor
        
        # Start with empty tables when not logged in
        self.clear_tables()
        
    def create_tab_bar(self):
        container = QWidget()
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Button container with center alignment
        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        button_layout.setContentsMargins(0, 0, 0, 20)  # Add bottom margin
        
        # Create styled buttons
        self.past_button = QPushButton("Past Events")
        self.today_button = QPushButton("Today's Events")
        
        # Style the buttons
        button_style = """
            QPushButton {
                padding: 10px 20px;
                font-size: 14px;
                border-radius: 15px;
                border: none;
                min-width: 150px;
            }
            QPushButton:checked {
                background-color: #4CAF50;
                color: white;
            }
            QPushButton:!checked {
                background-color: #e0e0e0;
                color: #333;
            }
        """
        self.past_button.setStyleSheet(button_style)
        self.today_button.setStyleSheet(button_style)
        
        # Make buttons checkable and exclusive
        self.past_button.setCheckable(True)
        self.today_button.setCheckable(True)
        button_group = QButtonGroup(self)
        button_group.addButton(self.past_button)
        button_group.addButton(self.today_button)
        
        # Center the buttons
        button_layout.addStretch()
        button_layout.addWidget(self.past_button)
        button_layout.addSpacing(20)  # Space between buttons
        button_layout.addWidget(self.today_button)
        button_layout.addStretch()
        
        # Create stacked widget for tables
        self.stack = QStackedWidget()
        
        # Create tables
        self.past_table = CalendarTable(self)
        self.today_table = CalendarTable(self)
        
        # Add tables to stack
        self.stack.addWidget(self.past_table)
        self.stack.addWidget(self.today_table)
        
        # Connect buttons to switch stacks
        self.past_button.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        self.today_button.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        
        # Set default to Today & Upcoming
        self.today_button.setChecked(True)
        self.stack.setCurrentIndex(1)  # Ensure today's view is shown by default
        
        main_layout.addWidget(button_container)
        main_layout.addWidget(self.stack)
        
        return container
    
    def create_top_bar(self):
        top_bar = QWidget()
        layout = QHBoxLayout(top_bar)
        
        # Cog icon button
        self.cog_btn = QPushButton("⚙")
        self.cog_btn.setFixedSize(40, 40)
        self.cog_btn.setStyleSheet("font-size: 16px; border: none; padding: 5px;")
        self.cog_btn.clicked.connect(self.show_settings_menu)
        
        layout.addWidget(self.cog_btn)
        layout.addStretch()
        
        return top_bar
    
    def create_info_bar(self):
        info_bar = QWidget()
        layout = QHBoxLayout(info_bar)
        
        # COG icon (left)
        self.cog_btn = QPushButton("⚙")
        self.cog_btn.setFixedSize(40, 40)
        self.cog_btn.setStyleSheet("font-size: 16px; border: none; padding: 5px;")
        self.cog_btn.clicked.connect(self.show_settings_menu)

        # User email (center)
        self.user_label = QLabel("Not logged in")
        self.user_label.setAlignment(Qt.AlignCenter)
        self.user_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        
        # Date (right)
        self.date_label = QLabel(self.current_date.strftime("%Y-%m-%d"))
        self.date_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        
        layout.addWidget(self.cog_btn)
        layout.addStretch()
        layout.addWidget(self.user_label)
        layout.addStretch()
        layout.addWidget(self.date_label)
        
        return info_bar
    
    def show_settings_menu(self):
        menu = QMenu(self)
        
        # Language submenu
        lang_menu = menu.addMenu("Interface Language")
        lang_menu.addAction("English", lambda: self.change_language("en"))
        lang_menu.addAction("日本語", lambda: self.change_language("ja"))
        
        # Speech recognition language submenu
        speech_menu = menu.addMenu("Speech Recognition")
        speech_menu.addAction("Auto-detect", lambda: self.change_speech_language("auto"))
        speech_menu.addAction("English", lambda: self.change_speech_language("en"))
        speech_menu.addAction("日本語", lambda: self.change_speech_language("ja"))
        
        # Auto-submit option
        self.auto_submit_action = menu.addAction("Auto-submit after speech")
        self.auto_submit_action.setCheckable(True)
        self.auto_submit_action.setChecked(getattr(self, 'auto_submit', False))
        self.auto_submit_action.triggered.connect(self.toggle_auto_submit)
        
        # Theme submenu
        theme_menu = menu.addMenu("Theme")
        theme_menu.addAction("Light", lambda: self.change_theme("light"))
        theme_menu.addAction("Dark", lambda: self.change_theme("dark"))
        
        if self.service:
            # Show these options only when logged in
            menu.addAction("Search by Date", self.search_by_date)
            menu.addAction("Add Event", self.add_event)
            menu.addSeparator()
            menu.addAction("Logout", self.logout)
        else:
            # Show login option when not logged in
            menu.addSeparator()
            menu.addAction("Login", self.show_login)
        
        # Show menu at cog button position
        button_pos = self.cog_btn.mapToGlobal(self.cog_btn.rect().bottomLeft())
        menu.exec_(button_pos)
    
    def show_login(self):
        login_dialog = LoginDialog(self)
        if login_dialog.exec_() == QDialog.Accepted:
            self.calendar_id = login_dialog.calendar_id
            self.user_email = login_dialog.user_email
            self.service = build('calendar', 'v3', credentials=login_dialog.credentials)
            self.user_label.setText(self.user_email)
            self.load_events()
            # Start the auto-refresh timer
            self.refresh_timer.start()
    
    def change_language(self, lang):
        self.language = lang
        # Save preference
        settings = QSettings("SEINX", "Calendar")
        settings.setValue("interface_language", lang)
        
        # Update UI text based on language
        self.update_ui_text()
        QMessageBox.information(
            self,
            "Language" if lang == "en" else "言語",
            "Language changed to " + ("English" if lang == "en" else "日本語")
        )
    
    def change_speech_language(self, lang):
        settings = QSettings("SEINX", "Calendar")
        settings.setValue("speech_language", lang)
        # Notify all speech widgets about the change
        for widget in self.findChildren(SpeechToTextWidget):
            widget.set_language(lang)
    
    def toggle_auto_submit(self, checked):
        settings = QSettings("SEINX", "Calendar")
        settings.setValue("auto_submit", checked)
        # Update all speech widgets
        for widget in self.findChildren(SpeechToTextWidget):
            widget.set_auto_submit(checked)
    
    def update_ui_text(self):
        # Update all UI text based on current language
        if self.language == "ja":
            self.setWindowTitle("SEINXカレンダー")
            self.past_button.setText("過去のイベント")
            self.today_button.setText("今日のイベント")
        else:
            self.setWindowTitle("SEINX Calendar")
            self.past_button.setText("Past Events")
            self.today_button.setText("Today's Events")
    
    def change_theme(self, theme):
        self.theme = theme
        self.apply_theme()
    
    def apply_theme(self):
        if self.theme == "dark":
            self.setStyleSheet("""
                QMainWindow { background-color: #2b2b2b; color: white; }
                QWidget { background-color: #2b2b2b; color: white; }
                QTabWidget::pane { background-color: #3c3c3c; }
                QTabBar::tab { background-color: #404040; color: white; padding: 8px; }
                QTabBar::tab:selected { background-color: #555555; }
                QTableWidget { background-color: #3c3c3c; alternate-background-color: #404040; }
                QHeaderView::section { background-color: #555555; color: white; }
                QPushButton { background-color: #555555; color: white; border: 1px solid #777; padding: 5px; }
                QPushButton:hover { background-color: #666666; }
            """)
        else:
            self.setStyleSheet("""
                QMainWindow { background-color: white; color: black; }
                QWidget { background-color: white; color: black; }
                QTabWidget::pane { background-color: #f0f0f0; }
                QTabBar::tab { background-color: #e0e0e0; color: black; padding: 8px; }
                QTabBar::tab:selected { background-color: #d0d0d0; }
                QTableWidget { background-color: white; alternate-background-color: #f5f5f5; }
                QHeaderView::section { background-color: #e0e0e0; color: black; }
                QPushButton { background-color: #e0e0e0; color: black; border: 1px solid #ccc; padding: 5px; }
                QPushButton:hover { background-color: #d0d0d0; }
            """)
    
    def search_by_date(self):
        dialog = DateSearchDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            selected_date = dialog.get_date()
            self.current_date = selected_date
            self.date_label.setText(selected_date.strftime("%Y-%m-%d"))
            self.load_events()
    
    def add_event(self):
        dialog = AddEventDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            event_data = dialog.get_event_data()
            self.create_calendar_event(event_data)
    
    def create_calendar_event(self, event_data):
        try:
            event = {
                'summary': event_data['name'],
                'location': event_data['location'],
                'description': event_data['remarks'],
            }
            
            # Handle all-day events differently
            if event_data.get('is_all_day'):
                event['start'] = {'date': event_data['start'].strftime('%Y-%m-%d')}
                event['end'] = {'date': (event_data['end'] + timedelta(days=1)).strftime('%Y-%m-%d')}
            else:
                event['start'] = {
                    'dateTime': event_data['start'].isoformat(),
                    'timeZone': 'UTC',
                }
                event['end'] = {
                    'dateTime': event_data['end'].isoformat(),
                    'timeZone': 'UTC',
                }
            
            event = self.service.events().insert(calendarId=self.calendar_id, body=event).execute()
            QMessageBox.information(self, "Success", "Event created successfully!")
            self.load_events()
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to create event: {str(e)}")
    
    def load_events(self):
        if not self.service:
            return
        
        try:
            # Get today's events
            today_start = datetime.combine(self.current_date, datetime.min.time())
            today_end = datetime.combine(self.current_date, datetime.max.time())
            
            today_events = self.get_events(today_start, today_end)
            
            # Get upcoming events (next 30 days)
            upcoming_end = today_end + timedelta(days=30)
            upcoming_events = self.get_events(today_end, upcoming_end)
            
            # Populate today's table with both today's and upcoming events
            self.populate_table(self.today_table, today_events, upcoming_events)
            
            # Get past events (last 30 days)
            past_start = today_start - timedelta(days=30)
            past_events = self.get_events(past_start, today_start)
            self.populate_table(self.past_table, past_events)
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load events: {str(e)}")
    
    def get_events(self, start_time, end_time):
        events_result = self.service.events().list(
            calendarId=self.calendar_id,
            timeMin=start_time.isoformat() + 'Z',
            timeMax=end_time.isoformat() + 'Z',
            singleEvents=True,
            orderBy='startTime',
            maxResults=2500,  # Get more events
            showDeleted=False  # Explicitly exclude deleted events
        ).execute()
        
        return events_result.get('items', [])
    
    def populate_table(self, table, events, upcoming_events=None):
        # Clear the table completely
        table.clearContents()
        table.event_data = {}  # Clear existing event data
        
        # Only show rows if logged in
        if not self.service:
            table.setRowCount(0)
            return
        
        # Calculate total rows needed
        total_rows = len(events)
        if upcoming_events:
            # Add 1 for separator and the length of upcoming events
            total_rows += len(upcoming_events) + 1
        
        # Set new row count (add extra rows for new events)
        table.setRowCount(total_rows + 20)
        
        # Filter out any deleted events
        active_events = [event for event in events if not event.get('status') == 'cancelled']
        
        for i, event in enumerate(active_events):
            start = event['start'].get('dateTime', event['start'].get('date'))
            end = event['end'].get('dateTime', event['end'].get('date'))
            
            # Parse datetime strings
            if 'T' in start:
                start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
                start_str = start_dt.strftime("%Y-%m-%d %H:%M")
            else:
                start_dt = datetime.fromisoformat(start)
                start_str = f"{start_dt.strftime('%Y-%m-%d')} (All day)"
            
            if 'T' in end:
                end_dt = datetime.fromisoformat(end.replace('Z', '+00:00'))
                end_str = end_dt.strftime("%Y-%m-%d %H:%M")
            else:
                end_dt = datetime.fromisoformat(end)
                end_str = f"{end_dt.strftime('%Y-%m-%d')} (All day)"
            
            # Create new items for each cell
            table.setItem(i, 0, QTableWidgetItem(event.get('summary', 'No Title')))
            table.setItem(i, 1, QTableWidgetItem(event.get('location', '')))
            table.setItem(i, 2, QTableWidgetItem(start_str))
            table.setItem(i, 3, QTableWidgetItem(end_str))
            table.setItem(i, 4, QTableWidgetItem(event.get('description', '')))
            
            # Store event data for this row
            table.event_data[i] = event
            
        current_row = len(active_events)
        
        # If we have upcoming events, add them after a separator
        if upcoming_events:
            # Add empty row before separator
            for col in range(5):
                empty_item = QTableWidgetItem("")
                table.setItem(current_row, col, empty_item)
            current_row += 1
            
            # Add separator row
            separator_item = QTableWidgetItem("Upcoming Events")
            separator_item.setBackground(QColor("#f0f0f0"))
            separator_item.setFont(QFont("Arial", 10, QFont.Bold))
            separator_item.setTextAlignment(Qt.AlignCenter)  # Center the text
            table.setItem(current_row, 0, separator_item)
            table.setSpan(current_row, 0, 1, 5)  # Merge all columns for the separator row
            separator_item.setFlags(separator_item.flags() & ~Qt.ItemIsEditable)  # Make it non-editable
            
            current_row += 1
            
            # Add upcoming events
            upcoming_active = [event for event in upcoming_events if not event.get('status') == 'cancelled']
            for event in upcoming_active:
                start = event['start'].get('dateTime', event['start'].get('date'))
                end = event['end'].get('dateTime', event['end'].get('date'))
                
                if 'T' in start:
                    start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
                    start_str = start_dt.strftime("%Y-%m-%d %H:%M")
                else:
                    start_dt = datetime.fromisoformat(start)
                    start_str = f"{start_dt.strftime('%Y-%m-%d')} (All day)"
                
                if 'T' in end:
                    end_dt = datetime.fromisoformat(end.replace('Z', '+00:00'))
                    end_str = end_dt.strftime("%Y-%m-%d %H:%M")
                else:
                    end_dt = datetime.fromisoformat(end)
                    end_str = f"{end_dt.strftime('%Y-%m-%d')} (All day)"
                
                table.setItem(current_row, 0, QTableWidgetItem(event.get('summary', 'No Title')))
                table.setItem(current_row, 1, QTableWidgetItem(event.get('location', '')))
                table.setItem(current_row, 2, QTableWidgetItem(start_str))
                table.setItem(current_row, 3, QTableWidgetItem(end_str))
                table.setItem(current_row, 4, QTableWidgetItem(event.get('description', '')))
                
                table.event_data[current_row] = event
                current_row += 1
        
        # Refresh the table
        table.viewport().update()
    
    def logout(self):
        # Stop the auto-refresh timer
        self.refresh_timer.stop()
        
        if os.path.exists('token.json'):
            os.remove('token.json')
        self.service = None
        self.user_email = ""
        self.calendar_id = None
        self.user_label.setText("No connected account")
        self.clear_tables()
        self.load_events()  # This will respect the logged-out state
    
    def clear_tables(self):
        # Clear and hide rows in all tables when logged out
        for table in [self.today_table, self.past_table]:
            table.clearContents()
            table.setRowCount(0)  # Set to 0 rows when logged out
            table.event_data = {}
    
    def update_event(self, event_data):
        dialog = UpdateEventDialog(event_data, self)
        if dialog.exec_() == QDialog.Accepted:
            updated_data = dialog.get_event_data()
            try:
                event = {
                    'summary': updated_data['name'],
                    'location': updated_data['location'],
                    'description': updated_data['remarks']
                }
                
                # Handle all-day events differently
                if updated_data.get('is_all_day'):
                    event['start'] = {'date': updated_data['start'].strftime('%Y-%m-%d')}
                    event['end'] = {'date': (updated_data['end'] + timedelta(days=1)).strftime('%Y-%m-%d')}
                else:
                    event['start'] = {
                        'dateTime': updated_data['start'].isoformat(),
                        'timeZone': 'UTC',
                    }
                    event['end'] = {
                        'dateTime': updated_data['end'].isoformat(),
                        'timeZone': 'UTC',
                    }
                
                # Preserve any existing fields that we don't update
                for key in event_data:
                    if key not in event and key not in ['start', 'end']:
                        event[key] = event_data[key]
                
                self.service.events().update(
                    calendarId=self.calendar_id,
                    eventId=event_data['id'],
                    body=event
                ).execute()
                
                QMessageBox.information(self, "Success", "Event updated successfully!")
                
                # Force a refresh from the server
                QTimer.singleShot(1000, self.load_events)  # Refresh after 1 second
                
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to update event: {str(e)}")
    
    def delete_event(self, event_data):
        reply = QMessageBox.question(
            self,
            "Delete Event",
            "Are you sure you want to delete this event?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                self.service.events().delete(
                    calendarId=self.calendar_id,
                    eventId=event_data['id']
                ).execute()
                
                QMessageBox.information(self, "Success", "Event deleted successfully!")
                
                # Force a refresh from the server
                QTimer.singleShot(1000, self.load_events)  # Refresh after 1 second
                
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to delete event: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())