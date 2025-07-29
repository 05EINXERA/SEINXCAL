# -----------------------------
# SEINXCAL: Calendar App with Voice Input
# -----------------------------
# Main application file
# -----------------------------

import sys
import os
import json
import shutil
import traceback
import threading
from datetime import datetime, timedelta

import numpy as np
import sounddevice as sd
from scipy.io import wavfile
import whisper
import qtawesome as qta

from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt, QDate, QDateTime, QTime, QEvent, QSettings, QPropertyAnimation, QEasingCurve
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTimeEdit, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QStackedWidget, QTableWidget, QTableWidgetItem, QDialog, QFormLayout, QLineEdit, QDateTimeEdit, QTextEdit, QMessageBox, QCheckBox, QDialogButtonBox, QAbstractItemView, QSizePolicy, QHeaderView, QButtonGroup, QMenu, QDesktopWidget, QComboBox, QShortcut, QDateEdit, QCompleter)
from PyQt5.QtGui import QFont, QIcon, QColor, QCursor, QKeySequence, QPainter
from PyQt5.QtCore import QStringListModel
from PyQt5.QtGui import QMovie

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import logging
from logging.handlers import RotatingFileHandler
import stat
import tzlocal
import pytz

# -----------------------------
# Logging Setup (with rotation)
# -----------------------------
LOG_FILE = 'seinxcalapp.log'
logger = logging.getLogger('seinxcalapp')
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(LOG_FILE, maxBytes=2*1024*1024, backupCount=3, encoding='utf-8')
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Global exception hook for uncaught exceptions
import sys
from PyQt5.QtWidgets import QMessageBox

def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    import traceback
    tb_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    logger.error(f"Uncaught exception: {tb_str}")
    QMessageBox.critical(None, "Unexpected Error", "An unexpected error occurred. Please check the log file for details.")

sys.excepthook = handle_exception

# -----------------------------
# Google Calendar API Scope
# -----------------------------
SCOPES = ['https://www.googleapis.com/auth/calendar']

# -----------------------------
# Token Management Class
# -----------------------------
class TokenManager:
    """
    Manages Google OAuth tokens including automatic refresh and persistence.
    Handles token validation, refresh, and storage.
    """
    
    def __init__(self):
        self.token_file = 'token.json'
        self.credentials = None
    
    def load_credentials(self):
        """Load credentials from token.json file."""
        try:
            if os.path.exists(self.token_file):
                self.credentials = Credentials.from_authorized_user_file(self.token_file, SCOPES)
                logger.info("Credentials loaded from token.json")
                return self.credentials
            else:
                logger.info("No token.json file found")
                return None
        except Exception as e:
            logger.error(f"Error loading credentials: {e}")
            return None
    
    def is_token_valid(self, credentials=None):
        """Check if the token is valid and not expired."""
        if credentials is None:
            credentials = self.credentials
        
        if not credentials:
            return False
        
        # Check if token is expired
        if credentials.expired:
            logger.info("Token is expired")
            return False
        
        # Check if we have a refresh token
        if not credentials.refresh_token:
            logger.warning("No refresh token available")
            return False
        
        return True
    
    def refresh_token_if_needed(self, credentials=None):
        """Refresh the token if it's expired."""
        if credentials is None:
            credentials = self.credentials
        
        if not credentials:
            return None
        
        try:
            if credentials.expired and credentials.refresh_token:
                logger.info("Refreshing expired token...")
                credentials.refresh(Request())
                logger.info("Token refreshed successfully")
                
                # Save the refreshed token
                self.save_credentials(credentials)
                return credentials
            elif credentials.expired and not credentials.refresh_token:
                logger.error("Token expired and no refresh token available")
                return None
            else:
                logger.info("Token is still valid")
                return credentials
        except Exception as e:
            logger.error(f"Error refreshing token: {e}")
            # Check if it's a specific error that we can handle
            if "invalid_grant" in str(e).lower():
                logger.error("Refresh token is invalid or revoked. User needs to re-authenticate.")
            elif "network" in str(e).lower() or "connection" in str(e).lower():
                logger.error("Network error during token refresh. Please check your internet connection.")
            return None
    
    def save_credentials(self, credentials=None):
        """Save credentials to token.json file."""
        if credentials is None:
            credentials = self.credentials
        
        if not credentials:
            logger.error("No credentials to save")
            return False
        
        try:
            with open(self.token_file, 'w') as token:
                token.write(credentials.to_json())
            logger.info("Credentials saved to token.json")
            return True
        except Exception as e:
            logger.error(f"Error saving credentials: {e}")
            return False
    
    def get_valid_credentials(self):
        """Get valid credentials, refreshing if necessary."""
        # Load credentials if not already loaded
        if not self.credentials:
            self.credentials = self.load_credentials()
        
        if not self.credentials:
            return None
        
        # Check if token is valid
        if self.is_token_valid():
            return self.credentials
        
        # Try to refresh the token
        refreshed_creds = self.refresh_token_if_needed()
        if refreshed_creds:
            self.credentials = refreshed_creds
            return self.credentials
        
        return None
    
    def get_token_info(self):
        """Get information about the current token."""
        if not self.credentials:
            return None
        
        info = {
            'expired': self.credentials.expired,
            'has_refresh_token': self.credentials.refresh_token is not None,
            'expiry': self.credentials.expiry.isoformat() if self.credentials.expiry else None,
            'scopes': self.credentials.scopes
        }
        
        if self.credentials.expiry:
            from datetime import datetime
            now = datetime.now(self.credentials.expiry.tzinfo)
            time_until_expiry = self.credentials.expiry - now
            info['seconds_until_expiry'] = time_until_expiry.total_seconds()
            info['expires_in_hours'] = time_until_expiry.total_seconds() / 3600
        
        return info
    
    def clear_credentials(self):
        """Clear stored credentials."""
        try:
            if os.path.exists(self.token_file):
                os.remove(self.token_file)
                logger.info("Credentials file removed")
            self.credentials = None
            return True
        except Exception as e:
            logger.error(f"Error clearing credentials: {e}")
            return False
    
    def create_new_credentials(self):
        """Create new credentials through OAuth flow."""
        try:
            if not os.path.exists('credentials.json'):
                raise FileNotFoundError("credentials.json file is missing")
            
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            credentials = flow.run_local_server(port=0)
            
            # Save the new credentials
            self.credentials = credentials
            self.save_credentials()
            
            logger.info("New credentials created and saved")
            return credentials
        except Exception as e:
            logger.error(f"Error creating new credentials: {e}")
            return None

# Global token manager instance
token_manager = TokenManager()

# -----------------------------
# Name Persistence Manager
# -----------------------------
class NamePersistenceManager:
    """
    Manages saving and loading of event names for autocomplete functionality.
    Saves names to a local text file and provides methods to add/retrieve names.
    """
    
    def __init__(self, filename='saved_names.txt'):
        self.filename = filename
        self.names = set()
        self.load_names()
    
    def load_names(self):
        """Load saved names from the text file."""
        try:
            if os.path.exists(self.filename):
                with open(self.filename, 'r', encoding='utf-8') as f:
                    names = f.read().strip().split('\n')
                    self.names = {name.strip() for name in names if name.strip()}
                logger.info(f"Loaded {len(self.names)} saved names from {self.filename}")
        except Exception as e:
            logger.error(f"Error loading names from {self.filename}: {e}")
            self.names = set()
    
    def save_names(self):
        """Save names to the text file."""
        try:
            with open(self.filename, 'w', encoding='utf-8') as f:
                for name in sorted(self.names):
                    f.write(f"{name}\n")
            logger.info(f"Saved {len(self.names)} names to {self.filename}")
        except Exception as e:
            logger.error(f"Error saving names to {self.filename}: {e}")
    
    def add_name(self, name):
        """Add a new name to the saved list."""
        if name and name.strip():
            name = name.strip()
            self.names.add(name)
            self.save_names()
            logger.info(f"Added name: {name}")
    
    def get_names(self):
        """Get all saved names as a sorted list."""
        return sorted(list(self.names))
    
    def get_names_starting_with(self, prefix):
        """Get names that start with the given prefix."""
        prefix = prefix.lower()
        return [name for name in self.names if name.lower().startswith(prefix)]
    
    def get_recent_names(self, count=3):
        """Get the most recent names (last added)."""
        return list(self.names)[-count:] if self.names else []
    
    def fuzzy_search(self, query, max_results=10):
        """Fuzzy search for names containing the query anywhere."""
        if not query:
            return []
        
        query = query.lower()
        results = []
        
        for name in self.names:
            name_lower = name.lower()
            # Check if query is contained anywhere in the name
            if query in name_lower:
                # Prioritize names that start with the query
                if name_lower.startswith(query):
                    results.insert(0, name)  # Add to beginning for priority
                else:
                    results.append(name)
        
        return results[:max_results]

# Global name persistence manager instance
name_manager = NamePersistenceManager()

# -----------------------------
# WhisperWorker: Handles audio recording and transcription
# -----------------------------
class WhisperWorker(QThread):
    """
    Worker thread for recording audio and transcribing it using OpenAI Whisper.
    Emits signals for finished transcription, errors, and status updates.
    """
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    status = pyqtSignal(str)
    
    def __init__(self, duration=5, parent=None):
        super().__init__(parent)
        self.duration = duration  # Recording duration in seconds
        self.sample_rate = 16000  # Whisper expects 16kHz
        self.temp_file = os.path.join(os.path.expanduser("~"), ".seinxcal_temp_new.wav")
        self.language = "en"  # Default language
        
        # Pre-import torch to check device availability
        try:
            import torch
            self.torch_available = True
            cuda_available = torch.cuda.is_available()
            self.device = 'cuda' if cuda_available else 'cpu'
            logger.info(f"[WhisperWorker] Device detection: {self.device}")
            logger.info(f"[WhisperWorker] Torch version: {torch.__version__}")
            logger.info(f"[WhisperWorker] CUDA available: {cuda_available}")
            if cuda_available:
                logger.info(f"[WhisperWorker] CUDA device count: {torch.cuda.device_count()}")
                logger.info(f"[WhisperWorker] CUDA device name: {torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'None'}")
        except ImportError:
            self.torch_available = False
            self.device = 'cpu'
            logger.warning("[WhisperWorker] PyTorch not available, using CPU fallback")
        except Exception as e:
            self.torch_available = False
            self.device = 'cpu'
            logger.error(f"[WhisperWorker] Error during device detection: {e}")
            logger.warning("[WhisperWorker] Using CPU fallback due to detection error")
    
    def set_language(self, lang):
        """Set the language for transcription."""
        self.language = lang
    
    def get_device_info(self):
        """Get detailed device information for debugging."""
        try:
            import torch
            info = {
                'torch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'device': self.device,
                'torch_available': self.torch_available
            }
            if torch.cuda.is_available():
                info['cuda_device_count'] = torch.cuda.device_count()
                info['cuda_device_name'] = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'None'
            return info
        except Exception as e:
            return {'error': str(e)}
    
    def force_cpu(self):
        """Force CPU usage for transcription."""
        self.device = 'cpu'
        logger.info("[WhisperWorker] Forced CPU usage")
    
    def run(self):
        try:
            # Check for ffmpeg
            logger.info(f"[WhisperWorker] Checking ffmpeg: {shutil.which('ffmpeg')}")
            if shutil.which("ffmpeg") is None:
                self.error.emit("ffmpeg is not installed or not in your PATH. Please install ffmpeg and try again.")
                logger.error("ffmpeg not found in PATH.")
                return
            
            # Check torch availability
            if not self.torch_available:
                self.error.emit("PyTorch is not available. Please install torch and try again.")
                logger.error("PyTorch not available for Whisper")
                return
            
            self.status.emit("Recording audio...")
            try:
                # Record audio from microphone
                logger.info(f"[WhisperWorker] Recording to temp file: {self.temp_file}")
                recording = sd.rec(int(self.duration * self.sample_rate), samplerate=self.sample_rate, channels=1, dtype='int16')
                sd.wait()
                wavfile.write(self.temp_file, self.sample_rate, recording)
                logger.info(f"[WhisperWorker] Temp file exists after recording: {os.path.exists(self.temp_file)}")
            except Exception as e:
                logger.error(f"[WhisperWorker] Audio recording failed: {e}\n{traceback.format_exc()}")
                self.error.emit(f"Audio recording failed: {e}")
                return
            
            self.status.emit("Transcribing...")
            try:
                # Import torch and whisper here to ensure they're available
                import torch
                import whisper
                
                # Transcribe using Whisper with proper device selection
                logger.info(f"[WhisperWorker] Checking temp file before transcription: {self.temp_file}, exists: {os.path.exists(self.temp_file)}")
                logger.info(f"[WhisperWorker] Using device: {self.device}")
                
                # Load model with explicit device specification
                model = whisper.load_model("base", device=self.device)
                result = model.transcribe(self.temp_file, language=self.language)
                text = result.get("text", "").strip()
                
                if not text:
                    self.error.emit("No speech detected. Please try again.")
                else:
                    self.finished.emit(text)
                    
            except Exception as e:
                logger.error(f"[WhisperWorker] Transcription failed: {e}\n{traceback.format_exc()}")
                # If CUDA fails, try CPU fallback
                if "cuda" in str(e).lower() and self.device == 'cuda':
                    logger.info("[WhisperWorker] CUDA failed, trying CPU fallback...")
                    self.status.emit("GPU failed, trying CPU...")
                    try:
                        import whisper
                        model = whisper.load_model("base", device='cpu')
                        result = model.transcribe(self.temp_file, language=self.language)
                        text = result.get("text", "").strip()
                        if text:
                            logger.info("[WhisperWorker] CPU fallback successful")
                            self.finished.emit(text)
                        else:
                            self.error.emit("No speech detected. Please try again.")
                    except Exception as cpu_error:
                        logger.error(f"[WhisperWorker] CPU fallback also failed: {cpu_error}")
                        self.error.emit(f"Transcription failed on both GPU and CPU: {str(e)}")
                elif "out of memory" in str(e).lower() and self.device == 'cuda':
                    logger.info("[WhisperWorker] CUDA out of memory, trying CPU fallback...")
                    self.status.emit("GPU memory full, trying CPU...")
                    try:
                        import whisper
                        model = whisper.load_model("base", device='cpu')
                        result = model.transcribe(self.temp_file, language=self.language)
                        text = result.get("text", "").strip()
                        if text:
                            logger.info("[WhisperWorker] CPU fallback successful after OOM")
                            self.finished.emit(text)
                        else:
                            self.error.emit("No speech detected. Please try again.")
                    except Exception as cpu_error:
                        logger.error(f"[WhisperWorker] CPU fallback also failed after OOM: {cpu_error}")
                        self.error.emit(f"Transcription failed on both GPU and CPU: {str(e)}")
                else:
                    self.error.emit(f"Transcription failed: {e}")
            finally:
                # Always clean up temp file
                if os.path.exists(self.temp_file):
                    try:
                        os.remove(self.temp_file)
                    except Exception as cleanup_err:
                        logger.warning(f"Failed to remove temp file: {cleanup_err}")
        except Exception as e:
            logger.error(f"[WhisperWorker] Unexpected error: {e}\n{traceback.format_exc()}")
            self.error.emit(str(e))

# -----------------------------
# SpeechToTextWidget: UI for voice input
# -----------------------------
class SpeechToTextWidget(QWidget):
    """
    Widget with a microphone button for voice input.
    Emits textCaptured when transcription is complete.
    """
    textCaptured = pyqtSignal(str)
    def __init__(self, parent=None, target_field=None):
        super().__init__(parent)
        self.target_field = target_field
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        self.mic_button = QPushButton()
        self.mic_button.setFixedSize(30, 30)  # Make it square
        if AppSettings.theme == 'dark':
            self.mic_button.setIcon(qta.icon('fa5s.microphone', color='white'))
        else:
            self.mic_button.setIcon(qta.icon('fa5s.microphone', color='black'))
        self.mic_button.setToolTip("Click to use voice input for this field")
        self.mic_button.clicked.connect(self.start_listening)
        
        # Style the button to be square and compact
        if AppSettings.theme == 'dark':
            self.mic_button.setStyleSheet("""
                QPushButton {
                    border: 1px solid #555;
                    border-radius: 4px;
                    background-color: #2c313a;
                    padding: 2px;
                }
                QPushButton:hover {
                    background-color: #3c414a;
                }
                QPushButton:pressed {
                    background-color: #1c212a;
                }
            """)
        else:
            self.mic_button.setStyleSheet("""
                QPushButton {
                    border: 1px solid #ccc;
                    border-radius: 4px;
                    background-color: #f8f9fa;
                    padding: 2px;
                }
                QPushButton:hover {
                    background-color: #e9ecef;
                }
                QPushButton:pressed {
                    background-color: #dee2e6;
                }
            """)
        
        layout.addWidget(self.mic_button)
        self.setLayout(layout)
        self.worker = None
        self.language = "en"
        self.auto_submit = False  # Default to manual submit
        # Use ListeningOverlay as a spinner overlay
        self.overlay = ListeningOverlay(self)
        self.overlay.hide()
    def set_language(self, lang):
        """Set the language for the next transcription."""
        self.language = lang
        if self.worker is not None:
            self.worker.set_language(lang)
    
    def set_auto_submit(self, enabled):
        """Set auto-submit mode for speech recognition."""
        self.auto_submit = enabled
    
    def force_cpu_usage(self):
        """Force CPU usage for debugging purposes."""
        if hasattr(self, 'worker') and self.worker:
            self.worker.force_cpu()
    
    def start_listening(self):
        """Start recording and transcribing audio."""
        # Disable the button first to prevent multiple clicks
        self.mic_button.setEnabled(False)
        
        # Ensure the overlay is properly positioned and shown
        if self.parent():
            # Force a layout update to ensure proper positioning
            self.parent().update()
            QApplication.processEvents()
        
        # Small delay to ensure UI stability before showing overlay
        QTimer.singleShot(50, self._show_overlay_and_start_worker)
    
    def _show_overlay_and_start_worker(self):
        """Show overlay and start the worker with a small delay."""
        # Show overlay with smooth animation
        self.overlay.show()
        self.overlay.raise_()  # Ensure it's on top
        
        # Create and start the worker
        self.worker = WhisperWorker()
        self.worker.set_language(self.language)
        self.worker.finished.connect(self.on_transcription_complete)
        self.worker.error.connect(self.on_transcription_error)
        self.worker.status.connect(self.on_status_update)
        self.worker.start()
    
    def on_transcription_complete(self, text):
        """Handle successful transcription."""
        # Hide overlay smoothly
        self.overlay.hide()
        
        # Re-enable the button
        self.mic_button.setEnabled(True)
        
        # Emit the captured text
        self.textCaptured.emit(text)
        
        # Update the target field if specified
        if self.target_field is not None:
            if isinstance(self.target_field, QTextEdit):
                self.target_field.append(text)
            else:
                self.target_field.setText(text)
    
    def on_transcription_error(self, error_msg):
        """Handle errors during transcription."""
        # Hide overlay smoothly
        self.overlay.hide()
        
        # Re-enable the button
        self.mic_button.setEnabled(True)
        
        # Show error message
        QMessageBox.warning(self, "Speech Recognition Error", error_msg)
    
    def on_status_update(self, status):
        """Update overlay status and button tooltip."""
        self.overlay.update_status(status)
        self.mic_button.setToolTip(status)
    
    def update_theme(self):
        """Update button styling when theme changes."""
        if AppSettings.theme == 'dark':
            self.mic_button.setIcon(qta.icon('fa5s.microphone', color='white'))
            self.mic_button.setStyleSheet("""
                QPushButton {
                    border: 1px solid #555;
                    border-radius: 4px;
                    background-color: #2c313a;
                    padding: 2px;
                }
                QPushButton:hover {
                    background-color: #3c414a;
                }
                QPushButton:pressed {
                    background-color: #1c212a;
                }
            """)
        else:
            self.mic_button.setIcon(qta.icon('fa5s.microphone', color='black'))
            self.mic_button.setStyleSheet("""
                QPushButton {
                    border: 1px solid #ccc;
                    border-radius: 4px;
                    background-color: #f8f9fa;
                    padding: 2px;
                }
                QPushButton:hover {
                    background-color: #e9ecef;
                }
                QPushButton:pressed {
                    background-color: #dee2e6;
                }
            """)
    
    def cleanup(self):
        """Clean up resources when widget is being destroyed."""
        if hasattr(self, 'overlay') and self.overlay:
            self.overlay.hide()
            self.overlay.deleteLater()
        if hasattr(self, 'worker') and self.worker:
            self.worker.quit()
            self.worker.wait()
            self.worker.deleteLater()
    
    def closeEvent(self, event):
        """Handle widget close event."""
        self.cleanup()
        super().closeEvent(event)

class ListeningOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Use Dialog instead of Popup to prevent flashing and ensure proper stacking
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)  # Don't steal focus
        self.setAttribute(Qt.WA_NoSystemBackground)  # Prevent system background
        
        # Create semi-transparent dark background with better styling
        self.setStyleSheet("""
            ListeningOverlay {
                background-color: rgba(0, 0, 0, 0.8);
                border-radius: 15px;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
            QLabel {
                color: white;
                background-color: transparent;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(15)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # Add microphone icon
        self.mic_label = QLabel()
        self.update_mic_icon()
        layout.addWidget(self.mic_label, alignment=Qt.AlignCenter)
        
        # Status label
        self.status_label = QLabel("Initializing...")
        self.set_status_label_color()
        self.status_label.setStyleSheet(self.status_label.styleSheet() + "\nfont-size: 16px; font-weight: bold;")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Progress indicator
        self.progress_label = QLabel()
        self.progress_label.setStyleSheet("font-size: 14px; color: #cccccc;")
        self.progress_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.progress_label)
        
        self.setLayout(layout)
        
        # Animation
        self.dots = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(500)
        
        # Set fixed size for the overlay
        self.setFixedSize(280, 180)
        
        self.current_status = ""
        
        # Add fade animation
        self.fade_anim = QPropertyAnimation(self, b"windowOpacity")
        self.fade_anim.setDuration(200)
        self.fade_anim.setEasingCurve(QEasingCurve.InOutQuad)
    
    def update_mic_icon(self, recording=False):
        if AppSettings.theme == 'dark':
            icon = qta.icon('fa5s.microphone' + ('-slash' if not recording else ''), color='white')
        else:
            color = '#4CAF50' if recording else 'gray'
            icon = qta.icon('fa5s.microphone' + ('-slash' if not recording else ''), color=color)
        self.mic_label.setPixmap(icon.pixmap(40, 40))  # Slightly larger icon
    
    def set_status_label_color(self):
        if AppSettings.theme == 'dark':
            self.status_label.setStyleSheet("color: white; font-size: 16px; font-weight: bold;")
        else:
            self.status_label.setStyleSheet("color: black; font-size: 16px; font-weight: bold;")
    
    def update_status(self, status):
        self.current_status = status
        self.status_label.setText(status)
        self.set_status_label_color()
        
        # Update microphone icon based on status
        if status == "Recording audio...":
            self.update_mic_icon(True)
        else:
            self.update_mic_icon(False)
    
    def update_animation(self):
        self.dots = (self.dots + 1) % 4
        self.progress_label.setText("." * self.dots)
    
    def showEvent(self, event):
        # Ensure proper positioning before showing
        if self.parent():
            # Try to find the main window for better centering
            main_window = self.parent()
            while main_window and not isinstance(main_window, QMainWindow):
                main_window = main_window.parent()
            
            if main_window:
                # Center on the main window
                window_rect = main_window.rect()
                window_center = main_window.mapToGlobal(window_rect.center())
                self.move(window_center.x() - self.width() // 2,
                         window_center.y() - self.height() // 2)
            else:
                # Fallback to parent widget centering
                parent_rect = self.parent().rect()
                parent_center = self.parent().mapToGlobal(parent_rect.center())
                self.move(parent_center.x() - self.width() // 2,
                         parent_center.y() - self.height() // 2)
        
        # Start with opacity 0 and fade in
        self.setWindowOpacity(0.0)
        super().showEvent(event)
        
        # Fade in animation
        self.fade_anim.setStartValue(0.0)
        self.fade_anim.setEndValue(1.0)
        self.fade_anim.start()
    
    def hideEvent(self, event):
        # Fade out before hiding
        self.fade_anim.setStartValue(1.0)
        self.fade_anim.setEndValue(0.0)
        self.fade_anim.finished.connect(super().hide)
        self.fade_anim.start()
        event.ignore()  # Don't hide immediately, let animation handle it
    
    def closeEvent(self, event):
        # Ensure proper cleanup
        self.timer.stop()
        self.fade_anim.stop()
        super().closeEvent(event)

# -----------------------------
# Security utilities
# -----------------------------
def set_secure_file_permissions(filepath):
    """Set file permissions to user-only (0600 on Unix, equivalent on Windows)."""
    try:
        if os.name == 'nt':
            import ctypes
            FILE_ATTRIBUTE_HIDDEN = 0x02
            ctypes.windll.kernel32.SetFileAttributesW(str(filepath), FILE_ATTRIBUTE_HIDDEN)
        else:
            os.chmod(filepath, stat.S_IRUSR | stat.S_IWUSR)
    except Exception as e:
        logger.warning(f"Could not set secure permissions on {filepath}: {e}")

class LoginDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr('login_title'))
        self.setFixedSize(400, 180)  # Increased height for status label
        self.calendar_id = None
        self.credentials = None
        self.user_email = None
        
        layout = QVBoxLayout()
        
        # Calendar ID input
        form_layout = QFormLayout()
        self.calendar_id_input = QLineEdit()
        self.calendar_id_input.setPlaceholderText(tr('calendar_id'))
        form_layout.addRow(tr('calendar_id'), self.calendar_id_input)
        layout.addLayout(form_layout)
        
        # Load last used calendar ID
        settings = QSettings("SEINX", "Calendar")
        last_calendar_id = settings.value("last_calendar_id", "")
        if last_calendar_id:
            self.calendar_id_input.setText(last_calendar_id)
        
        # Status label for auto-login feedback
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.status_label)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText(tr('ok'))
        buttons.button(QDialogButtonBox.Cancel).setText(tr('cancel'))
        buttons.accepted.connect(self.login)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.setLayout(layout)
        
        # Store the last calendar ID for potential auto-login
        self.last_calendar_id = last_calendar_id
        
    def try_auto_login(self):
        """Attempt to automatically log in using stored token and calendar ID."""
        try:
            logger.info("Attempting auto-login...")
            self.status_label.setText("Attempting auto-login...")
            QApplication.processEvents()  # Update UI
            
            # Use the stored calendar ID directly
            calendar_id = self.last_calendar_id
            
            # Get valid credentials using token manager
            creds = token_manager.get_valid_credentials()
            if creds:
                # Test the connection with stored calendar ID
                service = build('calendar', 'v3', credentials=creds)
                calendar = service.calendars().get(calendarId=calendar_id).execute()
                self.user_email = calendar.get('id', 'Unknown')
                self.credentials = creds
                self.calendar_id = calendar_id
                logger.info("Auto-login successful!")
                self.status_label.setText("Auto-login successful!")
                QApplication.processEvents()
                # Small delay to show success message
                QTimer.singleShot(1000, self.accept)
            else:
                logger.info("No valid credentials available for auto-login")
                self.status_label.setText("No valid credentials found")
        except Exception as e:
            logger.info(f"Auto-login failed: {e}")
            self.status_label.setText(f"Auto-login failed: {str(e)}")
            # Auto-login failed, user will need to manually log in
            pass
    
    def showEvent(self, event):
        """Override to handle auto-login after dialog is shown."""
        super().showEvent(event)
        # Try auto-login if we have both token and calendar ID
        if self.last_calendar_id and os.path.exists('token.json'):
            logger.info(f"Auto-login conditions met: calendar_id='{self.last_calendar_id}', token exists")
            # Show initial status
            self.status_label.setText("Checking stored credentials...")
            QApplication.processEvents()
            # Use a timer to delay the auto-login attempt slightly
            QTimer.singleShot(500, self.try_auto_login)  # Increased delay for better visibility
        else:
            logger.info(f"Auto-login conditions not met: calendar_id='{self.last_calendar_id}', token exists={os.path.exists('token.json')}")
            if not self.last_calendar_id:
                self.status_label.setText("No stored calendar ID found")
            elif not os.path.exists('token.json'):
                self.status_label.setText("No stored token found")
        
    def login(self):
        self.calendar_id = self.calendar_id_input.text().strip()
        if not self.calendar_id:
            QMessageBox.warning(self, tr('error'), tr('calendar_id'))
            return
        try:
            # Try to get valid credentials using token manager
            creds = token_manager.get_valid_credentials()
            
            # If no valid credentials, create new ones
            if not creds:
                if not os.path.exists('credentials.json'):
                    QMessageBox.critical(self, tr('error'), "credentials.json file is missing. Please place your Google API credentials file in the app directory.")
                    return
                
                creds = token_manager.create_new_credentials()
                if not creds:
                    QMessageBox.critical(self, tr('error'), "Failed to create new credentials. Please check your credentials.json file.")
                    return
            
            # Test the connection with provided calendar ID
            service = build('calendar', 'v3', credentials=creds)
            calendar = service.calendars().get(calendarId=self.calendar_id).execute()
            self.user_email = calendar.get('id', 'Unknown')
            self.credentials = creds
            
            # Save the calendar ID for future use
            settings = QSettings("SEINX", "Calendar")
            settings.setValue("last_calendar_id", self.calendar_id)
            
            self.accept()
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            QMessageBox.warning(self, tr('error'), f"{tr('event_failed')} {str(e)}")
            return

class DateSearchDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr('search_calendar_by_date'))
        self.setFixedSize(300, 150)
        
        layout = QVBoxLayout()
        
        label = QLabel(tr('select_date'))
        layout.addWidget(label)
        
        self.date_edit = QDateEdit()
        self.date_edit.setDate(QDate.currentDate())
        self.date_edit.setCalendarPopup(True)
        layout.addWidget(self.date_edit)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText(tr('ok'))
        buttons.button(QDialogButtonBox.Cancel).setText(tr('cancel'))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.setLayout(layout)
    
    def get_date(self):
        return self.date_edit.date().toPyDate()

class AddEventDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr('add_event_title'))
        self.setFixedSize(400, 320)
        
        layout = QVBoxLayout()
        
        # Event name with speech input and suggestions
        name_container = QWidget()
        name_layout = QHBoxLayout(name_container)
        name_layout.setContentsMargins(0, 0, 0, 0)
        name_layout.setSpacing(5)
        name_label = QLabel(tr('event_name'))
        name_label.setFixedWidth(80)  # Fixed width for label consistency
        
        self.name_edit = QLineEdit()
        
        # Create completer for autocomplete functionality
        self.name_completer = QCompleter()
        self.name_completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.name_completer.setCompletionMode(QCompleter.PopupCompletion)
        self.name_completer.setMaxVisibleItems(8)
        self.name_edit.setCompleter(self.name_completer)
        
        # Custom styling for modern input field
        if AppSettings.theme == 'dark':
            self.name_edit.setStyleSheet("""
                QLineEdit {
                    border: 1px solid #555;
                    border-radius: 4px;
                    padding: 5px;
                    background-color: #2c313a;
                    color: white;
                    min-height: 20px;
                }
            """)
        else:
            self.name_edit.setStyleSheet("""
                QLineEdit {
                    border: 1px solid #ccc;
                    border-radius: 4px;
                    padding: 5px;
                    background-color: white;
                    min-height: 20px;
                }
            """)
        
        # Connect text changed signal for dynamic suggestions
        self.name_edit.textChanged.connect(self.on_name_text_changed)
        
        self.name_speech = SpeechToTextWidget(target_field=self.name_edit)
        self.name_speech.textCaptured.connect(lambda text: self.name_edit.setText(text))
        
        name_layout.addWidget(name_label)
        name_layout.addWidget(self.name_edit, 1)  # Make it expand to fill available space
        name_layout.addWidget(self.name_speech)
        layout.addWidget(name_container)
        
        # Location with speech input
        location_container = QWidget()
        location_layout = QHBoxLayout(location_container)
        location_layout.setContentsMargins(0, 0, 0, 0)
        location_layout.setSpacing(5)
        location_label = QLabel(tr('location_label'))
        location_label.setFixedWidth(80)  # Fixed width for label consistency
        
        self.location_edit = QLineEdit()
        if AppSettings.theme == 'dark':
            self.location_edit.setStyleSheet("""
                QLineEdit {
                    border: 1px solid #555;
                    border-radius: 4px;
                    padding: 5px;
                    background-color: #2c313a;
                    color: white;
                    min-height: 20px;
                }
            """)
        else:
            self.location_edit.setStyleSheet("""
                QLineEdit {
                    border: 1px solid #ccc;
                    border-radius: 4px;
                    padding: 5px;
                    background-color: white;
                    min-height: 20px;
                }
            """)
        
        self.location_speech = SpeechToTextWidget(target_field=self.location_edit)
        self.location_speech.textCaptured.connect(lambda text: self.location_edit.setText(text))
        location_layout.addWidget(location_label)
        location_layout.addWidget(self.location_edit, 1)  # Make it expand to fill available space
        location_layout.addWidget(self.location_speech)
        layout.addWidget(location_container)
        
        # All Day checkbox
        self.all_day_check = QCheckBox(tr('all_day_event'))
        self.all_day_check.stateChanged.connect(self.on_all_day_changed)
        layout.addWidget(self.all_day_check)
        
        # Start date/time section
        start_container = QWidget()
        start_layout = QVBoxLayout(start_container)
        start_layout.setContentsMargins(0, 0, 0, 0)
        
        # Start date label
        start_date_label = QLabel(tr('start_datetime'))
        start_layout.addWidget(start_date_label)
        
        # Start date and time row
        start_datetime_row = QWidget()
        start_datetime_layout = QHBoxLayout(start_datetime_row)
        start_datetime_layout.setContentsMargins(0, 0, 0, 0)
        
        # Start date with weekday
        self.start_date = QDateEdit()
        self.start_date.setDate(QDate.currentDate())
        self.start_date.setCalendarPopup(True)
        self.start_date.setDisplayFormat("yyyy-MM-dd (ddd)")
        self.start_weekday_label = QLabel()
        self.start_weekday_label.setMinimumWidth(40)
        self.start_weekday_label.setAlignment(Qt.AlignCenter)
        self.update_start_weekday()
        self.start_date.dateChanged.connect(self.update_start_weekday)
        
        # Start time
        self.start_time = QTimeEdit()
        self.start_time.setTime(QTime.currentTime())
        self.start_time.setDisplayFormat("HH:mm")
        
        start_datetime_layout.addWidget(self.start_date)
        start_datetime_layout.addWidget(self.start_weekday_label)
        start_datetime_layout.addWidget(QLabel(tr('time')))
        start_datetime_layout.addWidget(self.start_time)
        start_layout.addWidget(start_datetime_row)
        layout.addWidget(start_container)
        
        # End date/time section
        end_container = QWidget()
        end_layout = QVBoxLayout(end_container)
        end_layout.setContentsMargins(0, 0, 0, 0)
        
        # End date label
        end_date_label = QLabel(tr('end_datetime'))
        end_layout.addWidget(end_date_label)
        
        # End date and time row
        end_datetime_row = QWidget()
        end_datetime_layout = QHBoxLayout(end_datetime_row)
        end_datetime_layout.setContentsMargins(0, 0, 0, 0)
        
        # End date with weekday
        self.end_date = QDateEdit()
        self.end_date.setDate(QDate.currentDate())
        self.end_date.setCalendarPopup(True)
        self.end_date.setDisplayFormat("yyyy-MM-dd (ddd)")
        self.end_weekday_label = QLabel()
        self.end_weekday_label.setMinimumWidth(40)
        self.end_weekday_label.setAlignment(Qt.AlignCenter)
        self.update_end_weekday()
        self.end_date.dateChanged.connect(self.update_end_weekday)
        
        # End time
        self.end_time = QTimeEdit()
        self.end_time.setTime(QTime.currentTime().addSecs(3600))
        self.end_time.setDisplayFormat("HH:mm")
        
        end_datetime_layout.addWidget(self.end_date)
        end_datetime_layout.addWidget(self.end_weekday_label)
        end_datetime_layout.addWidget(QLabel(tr('time')))
        end_datetime_layout.addWidget(self.end_time)
        end_layout.addWidget(end_datetime_row)
        layout.addWidget(end_container)
        
        # Remarks with speech input
        remarks_container = QWidget()
        remarks_layout = QHBoxLayout(remarks_container)
        remarks_layout.setContentsMargins(0, 0, 0, 0)
        remarks_label = QLabel(tr('remarks_label'))
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
        self.setTabOrder(self.name_edit, self.location_edit)
        self.setTabOrder(self.location_edit, self.start_date)
        self.setTabOrder(self.start_date, self.start_time)
        self.setTabOrder(self.start_time, self.end_date)
        self.setTabOrder(self.end_date, self.end_time)
        self.setTabOrder(self.end_time, self.remarks_edit)
        self.setTabOrder(self.remarks_edit, self.all_day_check)
        self.setTabOrder(self.all_day_check, self.name_speech)
        self.setTabOrder(self.name_speech, self.location_speech)
        self.setTabOrder(self.location_speech, self.remarks_speech)
        self.setTabOrder(self.remarks_speech, self.findChild(QDialogButtonBox))
        self.name_edit.setAccessibleName("Event Name")
        self.location_edit.setAccessibleName("Location")
        self.start_date.setAccessibleName("Start Date")
        self.start_time.setAccessibleName("Start Time")
        self.end_date.setAccessibleName("End Date")
        self.end_time.setAccessibleName("End Time")
        self.remarks_edit.setAccessibleName("Remarks")
        self.all_day_check.setAccessibleName("All Day Event Checkbox")
    
    def update_field_styling(self):
        """Update input field styling based on current theme."""
        if AppSettings.theme == 'dark':
            # Dark theme styling for name field
            self.name_edit.setStyleSheet("""
                QLineEdit {
                    border: 1px solid #555;
                    border-radius: 4px;
                    padding: 5px;
                    background-color: #2c313a;
                    color: white;
                    min-height: 20px;
                }
            """)
            # Dark theme styling for location field
            self.location_edit.setStyleSheet("""
                QLineEdit {
                    border: 1px solid #555;
                    border-radius: 4px;
                    padding: 5px;
                    background-color: #2c313a;
                    color: white;
                    min-height: 20px;
                }
            """)
        else:
            # Light theme styling for name field
            self.name_edit.setStyleSheet("""
                QLineEdit {
                    border: 1px solid #ccc;
                    border-radius: 4px;
                    padding: 5px;
                    background-color: white;
                    min-height: 20px;
                }
            """)
            # Light theme styling for location field
            self.location_edit.setStyleSheet("""
                QLineEdit {
                    border: 1px solid #ccc;
                    border-radius: 4px;
                    padding: 5px;
                    background-color: white;
                    min-height: 20px;
                }
            """)
        
        # Update microphone button styling too
        self.name_speech.update_theme()
        self.location_speech.update_theme()
        
        # Update weekday label styling
        self.update_start_weekday()
        self.update_end_weekday()
    
    def load_saved_names(self):
        """Load saved names into the completer for suggestions."""
        try:
            saved_names = name_manager.get_names()
            model = QStringListModel(saved_names)
            self.name_completer.setModel(model)
            logger.info(f"Loaded {len(saved_names)} saved names into completer")
        except Exception as e:
            logger.error(f"Error loading saved names: {e}")
    

    
    def on_name_text_changed(self, text):
        """Handle text changes to show dynamic suggestions with fuzzy search."""
        try:
            if not text:
                # If text is empty, show recent names
                recent_names = name_manager.get_recent_names(3)
                model = QStringListModel(recent_names)
                self.name_completer.setModel(model)
            else:
                # Use fuzzy search for better matching
                matching_names = name_manager.fuzzy_search(text, max_results=8)
                model = QStringListModel(matching_names)
                self.name_completer.setModel(model)
                
        except Exception as e:
            logger.error(f"Error in name suggestions: {e}")
    
    def showEvent(self, event):
        """Override to ensure field is empty when dialog is shown."""
        super().showEvent(event)
        # Only clear the field if it's empty (for new events)
        if not self.name_edit.text():
            # Show recent names when dialog opens for new events
            recent_names = name_manager.get_recent_names(3)
            model = QStringListModel(recent_names)
            self.name_completer.setModel(model)
    
    def update_start_weekday(self):
        """Update the start weekday label based on selected date."""
        date = self.start_date.date()
        weekday = date.dayOfWeek()
        # Qt returns 1-7 for Monday-Sunday, map to our translation keys
        weekday_map = {1: 'mon', 2: 'tue', 3: 'wed', 4: 'thu', 5: 'fri', 6: 'sat', 7: 'sun'}
        weekday_key = weekday_map.get(weekday, 'mon')
        self.start_weekday_label.setText(tr(weekday_key))
        
        # Update styling based on theme
        if AppSettings.theme == 'dark':
            self.start_weekday_label.setStyleSheet("""
                QLabel {
                    color: #4a9eff;
                    font-weight: bold;
                    padding: 2px;
                    border-radius: 3px;
                    background-color: #2c313a;
                }
            """)
        else:
            self.start_weekday_label.setStyleSheet("""
                QLabel {
                    color: #1976d2;
                    font-weight: bold;
                    padding: 2px;
                    border-radius: 3px;
                    background-color: #f5f5f5;
                }
            """)
    
    def update_end_weekday(self):
        """Update the end weekday label based on selected date."""
        date = self.end_date.date()
        weekday = date.dayOfWeek()
        # Qt returns 1-7 for Monday-Sunday, map to our translation keys
        weekday_map = {1: 'mon', 2: 'tue', 3: 'wed', 4: 'thu', 5: 'fri', 6: 'sat', 7: 'sun'}
        weekday_key = weekday_map.get(weekday, 'mon')
        self.end_weekday_label.setText(tr(weekday_key))
        
        # Update styling based on theme
        if AppSettings.theme == 'dark':
            self.end_weekday_label.setStyleSheet("""
                QLabel {
                    color: #4a9eff;
                    font-weight: bold;
                    padding: 2px;
                    border-radius: 3px;
                    background-color: #2c313a;
                }
            """)
        else:
            self.end_weekday_label.setStyleSheet("""
                QLabel {
                    color: #1976d2;
                    font-weight: bold;
                    padding: 2px;
                    border-radius: 3px;
                    background-color: #f5f5f5;
                }
            """)

    def on_all_day_changed(self, state):
        """Handle all-day event checkbox state change."""
        if state:
            # For all-day events, set times to start and end of day
            self.start_time.setTime(QTime(0, 0))
            # Set end time to 23:59:59 for proper all-day coverage
            self.end_time.setTime(QTime(23, 59, 59))
            # Hide time inputs for all-day events
            self.start_time.setVisible(False)
            self.end_time.setVisible(False)
            # Hide time labels
            for widget in self.findChildren(QLabel):
                if widget.text() == tr('time'):
                    widget.setVisible(False)
        else:
            # Show time inputs for non-all-day events
            self.start_time.setVisible(True)
            self.end_time.setVisible(True)
            # Show time labels
            for widget in self.findChildren(QLabel):
                if widget.text() == tr('time'):
                    widget.setVisible(True)
    
    def validate_input(self):
        """Validate event input and return (is_valid, error_message)."""
        # Validate event title
        event_name = self.name_edit.text().strip()
        if not event_name:
            return False, tr('event_title_required')
        
        # Validate start date/time
        start_date = self.start_date.date()
        if not start_date.isValid():
            return False, tr('invalid_start_date')
        
        # Validate end date/time
        end_date = self.end_date.date()
        if not end_date.isValid():
            return False, tr('invalid_end_date')
        
        # Validate date ranges (prevent extreme dates)
        current_year = QDate.currentDate().year()
        if start_date.year() < 1900 or start_date.year() > current_year + 10:
            return False, tr('start_date_out_of_range')
        if end_date.year() < 1900 or end_date.year() > current_year + 10:
            return False, tr('end_date_out_of_range')
        
        # Validate start and end time relationship
        is_all_day = self.all_day_check.isChecked()
        
        if is_all_day:
            # For all-day events, check if start date is before or equal to end date
            if start_date > end_date:
                return False, tr('start_date_after_end_date')
        else:
            # For timed events, check if start datetime is before end datetime
            start_dt = QDateTime(start_date, self.start_time.time())
            end_dt = QDateTime(end_date, self.end_time.time())
            
            if not start_dt.isValid():
                return False, tr('invalid_start_datetime')
            if not end_dt.isValid():
                return False, tr('invalid_end_datetime')
            
            if start_dt >= end_dt:
                return False, tr('start_time_after_end_time')
        
        return True, ""
    
    def accept(self):
        """Override accept to prevent dialog from closing when validation fails."""
        event_data = self.get_event_data()
        if event_data is not None:
            super().accept()
    
    def get_event_data(self):
        # Validate input first
        is_valid, error_message = self.validate_input()
        if not is_valid:
            QMessageBox.warning(self, tr('validation_error'), error_message)
            # Focus on the event name field if title is missing
            if not self.name_edit.text().strip():
                self.name_edit.setFocus()
                self.name_edit.selectAll()
            # Return None to keep dialog open
            return None
        
        is_all_day = self.all_day_check.isChecked()
        
        # Combine date and time for start datetime
        start_date = self.start_date.date()
        start_time = self.start_time.time()
        start_dt = QDateTime(start_date, start_time).toPyDateTime()
        
        # Combine date and time for end datetime
        end_date = self.end_date.date()
        end_time = self.end_time.time()
        end_dt = QDateTime(end_date, end_time).toPyDateTime()
        
        # Sanitize user input: strip, limit length, remove dangerous chars
        def sanitize(text, maxlen=256):
            return ''.join(c for c in text.strip() if c.isprintable())[:maxlen]
        
        event_name = sanitize(self.name_edit.text())
        
        # Save the name for future autocomplete
        if event_name:
            name_manager.add_name(event_name)
        
        return {
            'name': event_name,
            'location': sanitize(self.location_edit.text()),
            'start': start_dt.date() if is_all_day else start_dt,
            'end': end_dt.date() if is_all_day else end_dt,
            'remarks': sanitize(self.remarks_edit.toPlainText(), 1024),
            'is_all_day': is_all_day
        }

class CalendarTable(QTableWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_app = parent
        self.setColumnCount(5)
        self.setHorizontalHeaderLabels([
            tr('name'), tr('location'), tr('start_date'), tr('end_date'), tr('remarks')
        ])
        self.event_data = {}  # Store event data by row
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.viewport().installEventFilter(self)
        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Interactive)
        header.setSectionResizeMode(1, QHeaderView.Interactive)
        header.setSectionResizeMode(2, QHeaderView.Interactive)
        header.setSectionResizeMode(3, QHeaderView.Interactive)
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        total_width = self.viewport().width()
        self.setColumnWidth(0, int(total_width * 0.25))
        self.setColumnWidth(1, int(total_width * 0.25))
        self.setColumnWidth(2, int(total_width * 0.18))
        self.setColumnWidth(3, int(total_width * 0.18))
        # Remarks column (index 4) will adjust automatically due to Stretch mode
        # Disable default selection behavior to prevent interference with custom highlighting
        self.setSelectionMode(QAbstractItemView.NoSelection)
        self.setAlternatingRowColors(True)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.cellClicked.connect(self.handle_event_cell_click)
        self.actions_widget = None
        self.highlighted_row = None
        self.actions_timer = QTimer(self)
        self.actions_timer.setSingleShot(True)
        self.actions_timer.timeout.connect(self.hide_actions_widget)
        # Hide scroll bars
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # Hide row numbers
        self.verticalHeader().setVisible(False)
    def handle_event_cell_click(self, row, column):
        # Check if clicking on separator rows (don't highlight them)
        item = self.item(row, 0)
        if item and (item.data(Qt.UserRole) == 'date_separator' or item.data(Qt.UserRole) == 'breaker'):
            return  # Don't highlight separator rows
        
        # Check if clicking on the same row that's already highlighted
        if self.highlighted_row == row:
            # Toggle off - remove highlighting and hide action buttons
            self.hide_actions_widget()
            # Directly clear the highlight for this specific row
            for col in range(self.columnCount()):
                item = self.item(row, col)
                if item:
                    if AppSettings.theme == 'dark':
                        item.setBackground(QColor('#2c313a'))
                    else:
                        item.setBackground(QColor('white'))
            self.highlighted_row = None
            self.viewport().update()
            return
        
        # Highlight the clicked row with blue color
        self.clear_highlight()
        for col in range(self.columnCount()):
            item = self.item(row, col)
            if item:
                item.setBackground(QColor("#0078d4"))  # Blue highlight
        self.highlighted_row = row
        
        # Check if this is an empty row (no event data)
        item = self.item(row, 0)
        if item is None or item.text() == "":
            # Show add button for empty rows
            self.show_add_button(row)
            return
        if item.text() == "Upcoming Events":
            return
            
        # Show edit/delete actions for existing events
        if self.actions_widget:
            self.actions_widget.hide()
            self.actions_widget.deleteLater()
        self.show_actions_widget(row)
        self.setMouseTracking(True)
        # Keep actions visible for 5 seconds unless user clicks elsewhere
        self.actions_timer.start(5000)
    def show_add_button(self, row):
        """Show add button for empty rows."""
        if self.actions_widget:
            self.actions_widget.hide()
            self.actions_widget.deleteLater()
        
        self.actions_widget = QWidget(self)
        layout = QHBoxLayout(self.actions_widget)
        layout.setSpacing(3)
        layout.setContentsMargins(0, 0, 0, 0)
        
        add_btn = QPushButton(self.actions_widget)
        if AppSettings.theme == 'dark':
            add_icon = QIcon('icons/add.png') if os.path.exists('icons/add.png') else qta.icon('fa5s.plus', color='white')
        else:
            add_icon = QIcon('icons/add.png') if os.path.exists('icons/add.png') else qta.icon('fa5s.plus', color='black')
        
        add_btn.setIcon(add_icon)
        add_btn.setToolTip('Add Event')
        add_btn.setStyleSheet('border: none; background: transparent; color: white;')
        add_btn.setCursor(Qt.PointingHandCursor)
        add_btn.clicked.connect(self.parent_app.add_event)
        
        layout.addWidget(add_btn)
        
        rect = self.visualItemRect(self.item(row, 4))
        self.actions_widget.setFixedSize(40, rect.height()-2)
        horizontal_pos = rect.x() + rect.width() - 45
        vertical_pos = rect.y() - 1
        if horizontal_pos < rect.x():
            horizontal_pos = rect.x() + 5
        self.actions_widget.move(horizontal_pos, vertical_pos)
        self.actions_widget.show()
    
    def show_actions_widget(self, row):
        event_data = self.event_data.get(row)
        if not event_data:
            return
        self.actions_widget = QWidget(self)
        layout = QHBoxLayout(self.actions_widget)
        layout.setSpacing(3)
        layout.setContentsMargins(0, 0, 0, 0)
        edit_btn = QPushButton(self.actions_widget)
        if AppSettings.theme == 'dark':
            edit_icon = QIcon('icons/edit_white.png')
        else:
            edit_icon = QIcon.fromTheme('edit', QIcon('icons/edit.png'))
        edit_btn.setIcon(edit_icon)
        edit_btn.setToolTip('Edit')
        edit_btn.setStyleSheet('border: none; background: transparent; color: white;')
        edit_btn.setCursor(Qt.PointingHandCursor)
        edit_btn.clicked.connect(lambda: self.parent_app.update_event(event_data))
        delete_btn = QPushButton(self.actions_widget)
        if AppSettings.theme == 'dark':
            delete_icon = QIcon('icons/delete_white.png')
        else:
            delete_icon = QIcon.fromTheme('delete', QIcon('icons/delete.png'))
        delete_btn.setIcon(delete_icon)
        delete_btn.setToolTip('Delete')
        delete_btn.setStyleSheet('border: none; background: transparent; color: white;')
        delete_btn.setCursor(Qt.PointingHandCursor)
        delete_btn.clicked.connect(lambda: self.parent_app.delete_event(event_data))
        layout.addWidget(edit_btn)
        layout.addWidget(delete_btn)
        rect = self.visualItemRect(self.item(row, 4))
        self.actions_widget.setFixedSize(60, rect.height()-2)
        horizontal_pos = rect.x() + rect.width() - 65
        vertical_pos = rect.y() - 1
        if horizontal_pos < rect.x():
            horizontal_pos = rect.x() + 5
        self.actions_widget.move(horizontal_pos, vertical_pos)
        self.actions_widget.show()
    def hide_actions_widget(self):
        if self.actions_widget:
            self.actions_widget.hide()
            self.actions_widget.deleteLater()
            self.actions_widget = None
        # Stop the timer to prevent it from showing actions again
        self.actions_timer.stop()
        # Explicitly clear the highlight
        self.clear_highlight()
        # Force a repaint to ensure the highlight is removed
        self.viewport().update()
    def clear_highlight(self):
        if self.highlighted_row is not None:
            for col in range(self.columnCount()):
                item = self.item(self.highlighted_row, col)
                if item:
                    if AppSettings.theme == 'dark':
                        item.setBackground(QColor('#2c313a'))
                    else:
                        item.setBackground(QColor('white'))
            self.highlighted_row = None
    def leaveEvent(self, event):
        # Hide actions and highlight when mouse leaves the table or after timer
        if not self.underMouse():
            self.hide_actions_widget()
        super().leaveEvent(event)
    def mousePressEvent(self, event):
        # Hide actions if user clicks outside the highlighted row
        if self.highlighted_row is not None:
            row = self.rowAt(event.y())
            if row != self.highlighted_row:
                self.hide_actions_widget()
        super().mousePressEvent(event)
    def show_actions_menu(self, row, event_data):
        menu = QMenu(self)
        update_action = menu.addAction("Update")
        delete_action = menu.addAction("Delete")
        
        update_action.triggered.connect(lambda: self.parent_app.update_event(event_data))
        delete_action.triggered.connect(lambda: self.parent_app.delete_event(event_data))
        
        # Show menu at mouse position
        menu.exec_(QCursor.pos())
        
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
        self.setWindowTitle(tr('update_event_title'))
        
        # Pre-fill the fields with existing event data
        self.name_edit.setText(event_data.get('summary', ''))
        self.location_edit.setText(event_data.get('location', ''))
        self.remarks_edit.setText(event_data.get('description', ''))
        
        # First determine if this is an all-day event and set the checkbox
        # An event is all-day if it uses 'date' instead of 'dateTime'
        is_all_day = 'date' in event_data['start']
        self.all_day_check.setChecked(is_all_day)
        
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
        
        # Set the date and time separately
        self.start_date.setDate(QDate(start_dt.year, start_dt.month, start_dt.day))
        self.start_time.setTime(QTime(start_dt.hour, start_dt.minute))
        self.end_date.setDate(QDate(end_dt.year, end_dt.month, end_dt.day))
        self.end_time.setTime(QTime(end_dt.hour, end_dt.minute))
        
        # Update weekday labels
        self.update_start_weekday()
        self.update_end_weekday()
        
        # Handle all-day event visibility
        if is_all_day:
            self.start_time.setVisible(False)
            self.end_time.setVisible(False)
            # Hide time labels
            for widget in self.findChildren(QLabel):
                if widget.text() == tr('time'):
                    widget.setVisible(False)
    
    def get_event_data(self):
        """Override to ensure name persistence works for updates too."""
        event_data = super().get_event_data()
        # The parent class already handles name persistence
        return event_data

# -----------------------------
# SettingsDialog: Simple settings for language and theme
# -----------------------------
class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setFixedSize(300, 200)
        layout = QVBoxLayout()
        # Language selection
        lang_label = QLabel("Language:")
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["English", "日本語"])
        layout.addWidget(lang_label)
        layout.addWidget(self.lang_combo)
        # Theme selection
        theme_label = QLabel("Theme:")
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Light", "Dark"])
        layout.addWidget(theme_label)
        layout.addWidget(self.theme_combo)
        # Buttons
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)
        self.setLayout(layout)
    def get_settings(self):
        return {
            'language': self.lang_combo.currentText(),
            'theme': self.theme_combo.currentText()
        }

# -----------------------------
# MainWindow: Main application window
# -----------------------------
class SpinnerDialog(QDialog):
    def __init__(self, parent=None, message="Loading..."):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setModal(True)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("""
            QDialog {
                background-color: transparent;
            }
        """)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.spinner_label = QLabel(self)
        self.spinner_label.setAlignment(Qt.AlignCenter)
        self.spinner_label.setStyleSheet("background: transparent;")
        gif_path = os.path.join('icons', 'spinner.gif')
        if os.path.exists(gif_path):
            self.spinner_movie = QMovie(gif_path)
            self.spinner_label.setMovie(self.spinner_movie)
            self.spinner_movie.start()
        else:
            self.spinner_label.setText("⏳")
            self.spinner_label.setStyleSheet("font-size: 48px; color: white; background: transparent;")
        self.text_label = QLabel(message, self)
        self.text_label.setStyleSheet("color: white; font-size: 16px; margin-top: 12px; font-weight: bold; background: transparent;")
        self.text_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.spinner_label)
        layout.addWidget(self.text_label)
        self.setFixedSize(300, 200)
    
    def set_message(self, message):
        self.text_label.setText(message)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.service = None
        self.user_email = ""
        self.calendar_id = None
        self.current_date = datetime.now().date()
        self.theme = "light"
        self.language = "en"
        self.is_date_specific_view = False  # Track if we're showing a specific date
        
        # Set minimum size and get screen geometry
        self.setMinimumSize(1000, 600)
        screen = QDesktopWidget().availableGeometry()
        width = int(screen.width() * 0.8)  # 80% of screen width
        height = int(screen.height() * 0.8)  # 80% of screen height
        
        # Set size and center the window
        self.resize(width, height)
        self.move((screen.width() - width) // 2, (screen.height() - height) // 2)
        
        self.setWindowIcon(QIcon('icons/calendar-app-50.png'))
        self.setWindowTitle("SEINXCAL")
        
        # Setup auto-refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.force_table_refresh)
        self.refresh_timer.setInterval(30000)  # 30 seconds
        
        self.setup_ui()
        self.apply_theme()
        self.user_label.setText("No connected account")
        # Keyboard shortcuts
        self.shortcut_new = QShortcut(QKeySequence("Ctrl+N"), self)
        self.shortcut_new.activated.connect(self.add_event)
        self.shortcut_quit = QShortcut(QKeySequence("Ctrl+Q"), self)
        self.shortcut_quit.activated.connect(self.close)
        self.snackbar = Snackbar(self)
        
        # Auto-show login dialog on startup
        QTimer.singleShot(100, self.auto_show_login)
    
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
        self.past_button.clicked.connect(self.on_past_button_clicked)
        self.today_button.clicked.connect(self.on_today_button_clicked)
        
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

        # Today button (hidden by default)
        self.today_btn = QPushButton("Today")
        self.today_btn.setVisible(False)
        self.today_btn.setStyleSheet("font-size: 14px; padding: 4px 12px; border-radius: 8px;")
        self.today_btn.clicked.connect(self.reset_to_today)
        
        layout.addWidget(self.cog_btn)
        layout.addStretch()
        layout.addWidget(self.user_label)
        layout.addStretch()
        layout.addWidget(self.date_label)
        layout.addWidget(self.today_btn)
        
        return info_bar
    
    def show_settings_menu(self):
        menu = QMenu(self)
        # Language submenu at top level
        lang_menu = menu.addMenu(tr('language'))
        lang_menu.addAction(tr('english'), lambda: self.change_language('en'))
        lang_menu.addAction(tr('japanese'), lambda: self.change_language('ja'))
        # Speech recognition language submenu
        speech_menu = menu.addMenu(tr('speech_recognition'))
        speech_menu.addAction(tr('auto_detect'), lambda: self.change_speech_language('auto'))
        speech_menu.addAction(tr('english'), lambda: self.change_speech_language('en'))
        speech_menu.addAction(tr('japanese'), lambda: self.change_speech_language('ja'))
        # Auto-submit option
        self.auto_submit_action = menu.addAction(tr('auto_submit'))
        self.auto_submit_action.setCheckable(True)
        self.auto_submit_action.setChecked(getattr(self, 'auto_submit', False))
        self.auto_submit_action.triggered.connect(self.toggle_auto_submit)
        # Theme submenu
        theme_menu = menu.addMenu(tr('theme'))
        theme_menu.addAction(tr('light'), lambda: self.change_theme('light'))
        theme_menu.addAction(tr('dark'), lambda: self.change_theme('dark'))
        if self.service:
            menu.addAction(tr('search_by_date'), self.search_by_date)
            menu.addAction(tr('add_event'), self.add_event)
            menu.addSeparator()
            menu.addAction(tr('logout'), self.logout)
        else:
            menu.addSeparator()
            menu.addAction(tr('login'), self.show_login)
        button_pos = self.cog_btn.mapToGlobal(self.cog_btn.rect().bottomLeft())
        menu.exec_(button_pos)
    
    def auto_show_login(self):
        if not self.service:
            settings = QSettings("SEINX", "Calendar")
            last_calendar_id = settings.value("last_calendar_id", "")
            if last_calendar_id:
                spinner = SpinnerDialog(self, "Logging in with saved credentials...")
                def do_login():
                    try:
                        creds = token_manager.get_valid_credentials()
                        if creds:
                            service = build('calendar', 'v3', credentials=creds)
                            calendar = service.calendars().get(calendarId=last_calendar_id).execute()
                            self.calendar_id = last_calendar_id
                            self.user_email = calendar.get('id', 'Unknown')
                            self.service = service
                            calendar_name = calendar.get('summary', self.calendar_id)
                            self.user_label.setText(calendar_name)
                            self.load_events()
                            self.refresh_timer.start()
                            self.show_snackbar("Auto-login successful!", 2000)
                            spinner.accept()
                            return
                        spinner.accept()
                    except Exception as e:
                        logger.info(f"Silent auto-login failed: {e}")
                        spinner.accept()
                    self.show_login()
                # Give the spinner time to render and animate before starting login
                QTimer.singleShot(1000, do_login)
                spinner.exec_()
            else:
                self.show_login()
    
    def show_login(self):
        login_dialog = LoginDialog(self)
        if login_dialog.exec_() == QDialog.Accepted:
            self.calendar_id = login_dialog.calendar_id
            self.user_email = login_dialog.user_email
            self.service = build('calendar', 'v3', credentials=login_dialog.credentials)
            # Fetch and display calendar name
            try:
                calendar = self.service.calendars().get(calendarId=self.calendar_id).execute()
                calendar_name = calendar.get('summary', self.calendar_id)
                self.user_label.setText(calendar_name)
            except Exception as e:
                self.user_label.setText(self.calendar_id)
            self.load_events()
            self.refresh_timer.start()
    
    def change_language(self, lang):
        AppSettings.language = lang
        settings = QSettings("SEINX", "Calendar")
        settings.setValue("interface_language", lang)
        self.update_ui_text()
        self.update_all_labels_and_buttons()
        self.update_table_headers()
        self.update_date_format()
        # Notify all dialogs/tables to refresh language
        for widget in self.findChildren(QWidget):
            if hasattr(widget, 'refresh_language'):
                widget.refresh_language()
        # Show appropriate message based on language
        if lang == 'ja':
            message = '言語が日本語に変更されました'
        else:
            message = 'Language changed to English'
        QMessageBox.information(
            self,
            tr('language', lang),
            message
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
        if AppSettings.language == "ja":
            self.setWindowTitle("SEINXカレンダー")
            self.past_button.setText("過去のイベント")
            self.today_button.setText("今日のイベント")
            if hasattr(self, 'user_label'):
                if self.service:
                    try:
                        calendar = self.service.calendars().get(calendarId=self.calendar_id).execute()
                        calendar_name = calendar.get('summary', self.calendar_id)
                        self.user_label.setText(calendar_name)
                    except Exception:
                        self.user_label.setText(self.calendar_id or "ユーザー")
                else:
                    self.user_label.setText("未接続")
        else:
            self.setWindowTitle("SEINX Calendar")
            self.past_button.setText("Past Events")
            self.today_button.setText("Today's Events")
            if hasattr(self, 'user_label'):
                if self.service:
                    try:
                        calendar = self.service.calendars().get(calendarId=self.calendar_id).execute()
                        calendar_name = calendar.get('summary', self.calendar_id)
                        self.user_label.setText(calendar_name)
                    except Exception:
                        self.user_label.setText(self.calendar_id or "User")
                else:
                    self.user_label.setText("Not logged in")
    
    def update_all_labels_and_buttons(self):
        # Recursively update all labels and buttons to match the current language
        def update_widget_text(widget):
            if isinstance(widget, QLabel):
                if AppSettings.language == "ja":
                    if widget.text() == "過去のイベント":
                        widget.setText("過去のイベント")
                    elif widget.text() == "今日のイベント":
                        widget.setText("今日のイベント")
                else:
                    if widget.text() == "過去のイベント":
                        widget.setText("Past Events")
                    elif widget.text() == "今日のイベント":
                        widget.setText("Today's Events")
            elif isinstance(widget, QPushButton):
                if AppSettings.language == "ja":
                    if widget.text() == "イベント追加":
                        widget.setText("イベント追加")
                    elif widget.text() == "ログアウト":
                        widget.setText("ログアウト")
                else:
                    if widget.text() == "イベント追加":
                        widget.setText("Add Event")
                    elif widget.text() == "ログアウト":
                        widget.setText("Logout")
            for child in widget.findChildren(QWidget):
                update_widget_text(child)
        update_widget_text(self)
    
    def update_table_headers(self):
        # Update table headers for both tables
        if hasattr(self, 'today_table') and hasattr(self, 'past_table'):
            headers_en = ["Name", "Location", "Start Date", "End Date", "Remarks"]
            headers_ja = ["名前", "場所", "開始日", "終了日", "備考"]
            headers = headers_ja if AppSettings.language == "ja" else headers_en
            self.today_table.setHorizontalHeaderLabels(headers)
            self.past_table.setHorizontalHeaderLabels(headers)
    
    def update_date_format(self):
        # Update date label format
        if hasattr(self, 'date_label'):
            if AppSettings.language == "ja":
                self.date_label.setText(self.current_date.strftime("%Y/%m/%d"))
            else:
                self.date_label.setText(self.current_date.strftime("%Y-%m-%d"))
    
    def change_theme(self, theme):
        AppSettings.theme = theme
        self.theme = theme
        self.apply_theme()
        # Notify all tables to refresh theme and row backgrounds
        for widget in self.findChildren(CalendarTable):
            widget.viewport().update()
            # Always clear highlight so no row is left with the wrong color
            widget.clear_highlight()
    
    def apply_theme(self):
        if AppSettings.theme == "dark":
            self.setStyleSheet("""
                QMainWindow { background-color: #23272e; color: white; }
                QWidget { background-color: #2c313a; color: white; }
                QTabWidget::pane { background-color: #23272e; }
                QTabBar::tab { background-color: #2c313a; color: white; padding: 8px; }
                QTabBar::tab:selected { background-color: #3a3f4b; }
                QTableWidget { background-color: #23272e; alternate-background-color: #2c313a; }
                QHeaderView::section { background-color: #3a3f4b; color: white; }
                QPushButton { background-color: #3a3f4b; color: white; border: 1px solid #444a5a; padding: 5px; }
                QPushButton:hover { background-color: #4f5668; }
                QDialogButtonBox QPushButton { 
                    background-color: #3a3f4b; 
                    color: white; 
                    border: 1px solid #444a5a; 
                    padding: 8px 16px; 
                    border-radius: 4px;
                    min-width: 80px;
                }
                QDialogButtonBox QPushButton:hover { 
                    background-color: #4f5668; 
                    border-color: #555a6a;
                }
                QDialogButtonBox QPushButton:pressed { 
                    background-color: #2c313a; 
                    border-color: #3a3f4b;
                }
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
                QDialogButtonBox QPushButton { 
                    background-color: #e0e0e0; 
                    color: black; 
                    border: 1px solid #ccc; 
                    padding: 8px 16px; 
                    border-radius: 4px;
                    min-width: 80px;
                }
                QDialogButtonBox QPushButton:hover { 
                    background-color: #d0d0d0; 
                    border-color: #adb5bd;
                }
                QDialogButtonBox QPushButton:pressed { 
                    background-color: #c0c0c0; 
                    border-color: #a0a0a0;
                }
            """)
    
    def search_by_date(self):
        dialog = DateSearchDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            selected_date = dialog.get_date()
            self.current_date = selected_date
            self.date_label.setText(selected_date.strftime("%Y-%m-%d"))
            self.is_date_specific_view = True  # Set flag for date-specific view
            self.load_events_for_specific_date(selected_date)
            # Show Today button if not today
            if self.current_date != datetime.now().date():
                self.today_btn.setVisible(True)
            else:
                self.today_btn.setVisible(False)
    
    def add_event(self):
        if self.is_date_specific_view:
            # Show message that add event is disabled in date search mode
            QMessageBox.information(
                self, 
                tr('add_event'), 
                tr('add_disabled_in_search')
            )
            return
        
        dialog = AddEventDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            event_data = dialog.get_event_data()
            if event_data:  # Only create event if validation passed
                self.create_calendar_event(event_data)
    
    def create_calendar_event(self, event_data):
        try:
            import tzlocal
            local_tz = tzlocal.get_localzone_name()
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
                    'timeZone': local_tz,
                }
                event['end'] = {
                    'dateTime': event_data['end'].isoformat(),
                    'timeZone': local_tz,
                }
            
            event = self.service.events().insert(calendarId=self.calendar_id, body=event).execute()
            self.show_snackbar(tr('event_created'))
            self.force_table_refresh()
            
        except Exception as e:
            QMessageBox.warning(self, tr('error'), f"{tr('event_failed')} {str(e)}")
    
    def load_events_for_specific_date(self, target_date):
        """Load events only for the specific date, without past or upcoming events."""
        if not self.service:
            return
        
        try:
            # Get local timezone using tzlocal
            local_tz = tzlocal.get_localzone()
            
            # Create QDateTime objects for start and end of the selected day
            selected_qdate = QDate(target_date.year, target_date.month, target_date.day)
            start_datetime_q = QDateTime(selected_qdate, QTime(0, 0, 0))
            end_datetime_q = QDateTime(selected_qdate.addDays(1), QTime(0, 0, 0))
            
            # Convert QDateTime to timezone-aware Python datetime objects
            start_datetime = start_datetime_q.toPyDateTime()
            end_datetime = end_datetime_q.toPyDateTime()
            
            # Make them timezone-aware using local timezone
            # Check if local_tz is pytz or zoneinfo and handle accordingly
            if hasattr(local_tz, 'localize'):
                # pytz timezone
                start_datetime_local = local_tz.localize(start_datetime)
                end_datetime_local = local_tz.localize(end_datetime)
            else:
                # zoneinfo timezone
                start_datetime_local = start_datetime.replace(tzinfo=local_tz)
                end_datetime_local = end_datetime.replace(tzinfo=local_tz)
            
            # Convert to UTC for Google Calendar API
            start_datetime_utc = start_datetime_local.astimezone(pytz.utc)
            end_datetime_utc = end_datetime_local.astimezone(pytz.utc)
            
            # Format as ISO 8601 with UTC timezone for Google API
            time_min = start_datetime_utc.isoformat()
            time_max = end_datetime_utc.isoformat()
            
            # Get events using the correctly calculated time range
            date_events = self.get_events_with_timerange(time_min, time_max)
            
            # Populate the currently active table with the date-specific events
            date_str = target_date.strftime("%Y-%m-%d")
            custom_title = tr('events_for_date').format(date=date_str)
            
            # Determine which table is currently visible and populate it
            current_index = self.stack.currentIndex()
            if current_index == 0:  # Past Events tab
                self.populate_table(self.past_table, date_events, custom_title=custom_title)
                # Clear the other table
                self.today_table.clearContents()
                self.today_table.clearSpans()
                self.today_table.setRowCount(0)
                self.today_table.event_data = {}
            else:  # Today's Events tab (index 1)
                self.populate_table(self.today_table, date_events, custom_title=custom_title)
                # Clear the other table
                self.past_table.clearContents()
                self.past_table.clearSpans()
                self.past_table.setRowCount(0)
                self.past_table.event_data = {}
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load events for date: {str(e)}")
    
    def categorize_events(self, events, today_start, today_end):
        """Categorize events into today's events and upcoming events without duplication."""
        today_events = []
        upcoming_events = []
        
        for event in events:
            if event.get('status') == 'cancelled':
                continue
                
            # Parse event start and end times
            start_data = event['start']
            end_data = event['end']
            
            # Handle all-day events
            if 'date' in start_data:
                # All-day event
                start_date = datetime.fromisoformat(start_data['date']).date()
                end_date = datetime.fromisoformat(end_data['date']).date()
                
                # Check if event spans today
                event_spans_today = (start_date <= today_start.date() and 
                                   end_date > today_start.date())
                
                if event_spans_today:
                    today_events.append(event)
                elif start_date > today_end.date():
                    upcoming_events.append(event)
                    
            else:
                # Timed event
                start_dt = datetime.fromisoformat(start_data['dateTime'].replace('Z', '+00:00'))
                end_dt = datetime.fromisoformat(end_data['dateTime'].replace('Z', '+00:00'))
                
                # Convert to local timezone for comparison
                local_tz = tzlocal.get_localzone()
                if hasattr(local_tz, 'localize'):
                    start_local = local_tz.localize(start_dt.replace(tzinfo=None))
                    end_local = local_tz.localize(end_dt.replace(tzinfo=None))
                else:
                    start_local = start_dt.replace(tzinfo=local_tz)
                    end_local = end_dt.replace(tzinfo=local_tz)
                
                # Check if event is today's event
                is_today_event = (
                    start_local < today_end and end_local > today_start
                )
                
                if is_today_event:
                    today_events.append(event)
                elif start_local >= today_end:
                    upcoming_events.append(event)
        
        return today_events, upcoming_events
    
    def load_events(self):
        if not self.service:
            return
        
        try:
            # Get local timezone using tzlocal
            local_tz = tzlocal.get_localzone()
            
            # Define today's boundaries
            today_qdate = QDate(self.current_date.year, self.current_date.month, self.current_date.day)
            today_start_q = QDateTime(today_qdate, QTime(0, 0, 0))
            today_end_q = QDateTime(today_qdate.addDays(1), QTime(0, 0, 0))
            
            # Convert to timezone-aware datetime objects
            if hasattr(local_tz, 'localize'):
                today_start = local_tz.localize(today_start_q.toPyDateTime())
                today_end = local_tz.localize(today_end_q.toPyDateTime())
            else:
                today_start = today_start_q.toPyDateTime().replace(tzinfo=local_tz)
                today_end = today_end_q.toPyDateTime().replace(tzinfo=local_tz)
            
            # Fetch broader range of events (today + next 30 days)
            today_start_utc = today_start.astimezone(pytz.utc)
            upcoming_end_q = QDateTime(today_qdate.addDays(31), QTime(0, 0, 0))
            if hasattr(local_tz, 'localize'):
                upcoming_end = local_tz.localize(upcoming_end_q.toPyDateTime())
            else:
                upcoming_end = upcoming_end_q.toPyDateTime().replace(tzinfo=local_tz)
            upcoming_end_utc = upcoming_end.astimezone(pytz.utc)
            
            # Fetch all events in the range
            all_events = self.get_events_with_timerange(
                today_start_utc.isoformat(), 
                upcoming_end_utc.isoformat()
            )
            
            # Categorize events without duplication
            today_events, upcoming_events = self.categorize_events(
                all_events, today_start, today_end
            )
            
            # Populate today's table with properly categorized events
            self.populate_table(self.today_table, today_events, upcoming_events)
            
            # Get past events (last 30 days)
            past_start_q = QDateTime(today_qdate.addDays(-30), QTime(0, 0, 0))
            if hasattr(local_tz, 'localize'):
                past_start = local_tz.localize(past_start_q.toPyDateTime())
            else:
                past_start = past_start_q.toPyDateTime().replace(tzinfo=local_tz)
            past_start_utc = past_start.astimezone(pytz.utc)
            time_min_past = past_start_utc.isoformat()
            
            past_events = self.get_events_with_timerange(time_min_past, today_start_utc.isoformat())
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
    
    def get_events_with_timerange(self, time_min, time_max):
        """Get events using pre-formatted timeMin and timeMax strings."""
        events_result = self.service.events().list(
            calendarId=self.calendar_id,
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy='startTime',
            maxResults=2500,  # Get more events
            showDeleted=False  # Explicitly exclude deleted events
        ).execute()
        
        return events_result.get('items', [])
    
    def format_date_with_weekday(self, dt, include_time=True, is_all_day=False):
        """Format a datetime object to include weekday."""
        # Get weekday name
        weekday_map = {0: 'mon', 1: 'tue', 2: 'wed', 3: 'thu', 4: 'fri', 5: 'sat', 6: 'sun'}
        weekday = dt.weekday()  # 0=Monday, 6=Sunday
        weekday_key = weekday_map.get(weekday, 'mon')
        weekday_name = tr(weekday_key)
        
        if is_all_day:
            return f"{dt.strftime('%Y-%m-%d')} ({weekday_name}) ({tr('all_day')})"
        elif include_time:
            return f"{dt.strftime('%Y-%m-%d')} ({weekday_name}) {dt.strftime('%H:%M')}"
        else:
            return f"{dt.strftime('%Y-%m-%d')} ({weekday_name})"
    
    def populate_table(self, table, events, upcoming_events=None, custom_title=None):
        """Populate table with events, ensuring proper row structure."""
        # Clear the table completely and reset structure
        table.clearContents()
        table.clearSpans()  # Clear any merged cells
        table.event_data = {}  # Clear existing event data
        
        # Reset table structure completely
        table.setRowCount(0)
        
        # Only show rows if logged in
        if not self.service:
            return
        
        # Filter out any deleted events
        active_events = [event for event in events if not event.get('status') == 'cancelled']
        upcoming_active = [event for event in upcoming_events if not event.get('status') == 'cancelled'] if upcoming_events else []
        
        # Calculate total rows needed
        total_rows = len(active_events)
        if custom_title:
            total_rows += 1  # Add 1 for custom title separator
        if upcoming_events:
            # Add 2 for separator (empty row + separator row) and the length of upcoming events
            total_rows += len(upcoming_active) + 2
        
        # Set new row count
        table.setRowCount(total_rows)
        
        current_row = 0
        
        # Add custom title separator if provided
        if custom_title:
            # Add separator row with custom title
            separator_item = QTableWidgetItem(custom_title)
            if AppSettings.theme == 'dark':
                separator_item.setBackground(QColor("#2c313a"))
                separator_item.setForeground(QColor("#4a9eff"))
            else:
                separator_item.setBackground(QColor("#f8f9fa"))
                separator_item.setForeground(QColor("#1976d2"))
            separator_item.setData(Qt.UserRole, 'date_separator')
            separator_item.setFont(QFont("Arial", 10, QFont.Bold))
            separator_item.setTextAlignment(Qt.AlignCenter)
            table.setItem(current_row, 0, separator_item)
            table.setSpan(current_row, 0, 1, 5)  # Merge all columns for the separator row
            separator_item.setFlags(separator_item.flags() & ~Qt.ItemIsEditable)
            current_row += 1
        
        # Add main events
        for event in active_events:
            start = event['start'].get('dateTime', event['start'].get('date'))
            end = event['end'].get('dateTime', event['end'].get('date'))
            
            # Parse datetime strings
            if 'T' in start:
                start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
                start_str = self.format_date_with_weekday(start_dt, include_time=True, is_all_day=False)
            else:
                start_dt = datetime.fromisoformat(start)
                start_str = self.format_date_with_weekday(start_dt, include_time=False, is_all_day=True)
            
            if 'T' in end:
                end_dt = datetime.fromisoformat(end.replace('Z', '+00:00'))
                end_str = self.format_date_with_weekday(end_dt, include_time=True, is_all_day=False)
            else:
                end_dt = datetime.fromisoformat(end)
                end_str = self.format_date_with_weekday(end_dt, include_time=False, is_all_day=True)
            
            # Create new items for each cell
            table.setItem(current_row, 0, QTableWidgetItem(event.get('summary', 'No Title')))
            table.setItem(current_row, 1, QTableWidgetItem(event.get('location', '')))
            table.setItem(current_row, 2, QTableWidgetItem(start_str))
            table.setItem(current_row, 3, QTableWidgetItem(end_str))
            table.setItem(current_row, 4, QTableWidgetItem(event.get('description', '')))
            
            # Store event data for this row
            table.event_data[current_row] = event
            current_row += 1
        
        # If we have upcoming events, add them after a separator
        if upcoming_events and upcoming_active:
            # Add empty row before separator (only if we don't have a custom title)
            if not custom_title:
                for col in range(5):
                    empty_item = QTableWidgetItem("")
                    table.setItem(current_row, col, empty_item)
                current_row += 1
            
            # Add separator row
            separator_item = QTableWidgetItem(tr('upcoming_events'))
            if AppSettings.theme == 'dark':
                separator_item.setBackground(QColor("#333333"))
                separator_item.setForeground(QColor("#ffffff"))
            else:
                separator_item.setBackground(QColor("#f0f0f0"))
                separator_item.setForeground(QColor("#222222"))
            separator_item.setData(Qt.UserRole, 'breaker')
            separator_item.setFont(QFont("Arial", 10, QFont.Bold))
            separator_item.setTextAlignment(Qt.AlignCenter)
            table.setItem(current_row, 0, separator_item)
            table.setSpan(current_row, 0, 1, 5)  # Merge all columns for the separator row
            separator_item.setFlags(separator_item.flags() & ~Qt.ItemIsEditable)
            current_row += 1
            
            # Add upcoming events
            for event in upcoming_active:
                start = event['start'].get('dateTime', event['start'].get('date'))
                end = event['end'].get('dateTime', event['end'].get('date'))
                
                if 'T' in start:
                    start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
                    start_str = self.format_date_with_weekday(start_dt, include_time=True, is_all_day=False)
                else:
                    start_dt = datetime.fromisoformat(start)
                    start_str = self.format_date_with_weekday(start_dt, include_time=False, is_all_day=True)
                
                if 'T' in end:
                    end_dt = datetime.fromisoformat(end.replace('Z', '+00:00'))
                    end_str = self.format_date_with_weekday(end_dt, include_time=True, is_all_day=False)
                else:
                    end_dt = datetime.fromisoformat(end)
                    end_str = self.format_date_with_weekday(end_dt, include_time=False, is_all_day=True)
                
                table.setItem(current_row, 0, QTableWidgetItem(event.get('summary', 'No Title')))
                table.setItem(current_row, 1, QTableWidgetItem(event.get('location', '')))
                table.setItem(current_row, 2, QTableWidgetItem(start_str))
                table.setItem(current_row, 3, QTableWidgetItem(end_str))
                table.setItem(current_row, 4, QTableWidgetItem(event.get('description', '')))
                
                table.event_data[current_row] = event
                current_row += 1
        
        # Add empty rows for better UX
        visible_rows = table.viewport().height() // table.rowHeight(0) if table.rowCount() > 0 else 10
        extra_rows = max(0, visible_rows - table.rowCount())
        for _ in range(extra_rows):
            row = table.rowCount()
            table.insertRow(row)
            for col in range(table.columnCount()):
                table.setItem(row, col, QTableWidgetItem(""))
        
        # Refresh the table
        table.viewport().update()
    
    def on_past_button_clicked(self):
        """Handle past button click - reset to normal view if in date-specific mode."""
        if self.is_date_specific_view:
            self.reset_to_today()
            self.stack.setCurrentIndex(0)  # Switch to past events table
        else:
            self.stack.setCurrentIndex(0)
    
    def on_today_button_clicked(self):
        """Handle today button click - reset to normal view if in date-specific mode."""
        if self.is_date_specific_view:
            self.reset_to_today()
        else:
            self.stack.setCurrentIndex(1)
    
    def reset_to_today(self):
        self.current_date = datetime.now().date()
        self.date_label.setText(self.current_date.strftime("%Y-%m-%d"))
        self.is_date_specific_view = False  # Clear flag for regular view
        self.load_events()
        self.today_btn.setVisible(False)
    
    def logout(self):
        # Stop the auto-refresh timer
        self.refresh_timer.stop()
        
        # Clear credentials using token manager
        token_manager.clear_credentials()
        
        # Clear stored calendar ID
        settings = QSettings("SEINX", "Calendar")
        settings.remove("last_calendar_id")
        
        self.service = None
        self.user_email = ""
        self.calendar_id = None
        self.user_label.setText("No connected account")
        self.clear_tables()
        # No need to refresh when logged out
    
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
                import tzlocal
                local_tz = tzlocal.get_localzone_name()
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
                        'timeZone': local_tz,
                    }
                    event['end'] = {
                        'dateTime': updated_data['end'].isoformat(),
                        'timeZone': local_tz,
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
                self.show_snackbar(tr('event_update_success'))
                
                # Force an immediate refresh from the server
                self.force_table_refresh()
                
            except Exception as e:
                QMessageBox.warning(self, tr('error'), f"{tr('event_update_failed')} {str(e)}")
    
    def delete_event(self, event_data):
        reply = QMessageBox.question(
            self,
            tr('delete_event'),
            tr('delete_confirm'),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                self.service.events().delete(
                    calendarId=self.calendar_id,
                    eventId=event_data['id']
                ).execute()
                self.show_snackbar(tr('event_deleted'))
                
                # Force an immediate refresh from the server
                self.force_table_refresh()
                
            except Exception as e:
                QMessageBox.warning(self, tr('error'), f"{tr('event_failed')} {str(e)}")

    def show_settings_dialog(self):
        dlg = SettingsDialog(self)
        # Set current values
        if AppSettings.language == "ja":
            dlg.lang_combo.setCurrentIndex(1)
        else:
            dlg.lang_combo.setCurrentIndex(0)
        dlg.theme_combo.setCurrentIndex(1 if AppSettings.theme == "dark" else 0)
        if dlg.exec_() == QDialog.Accepted:
            settings = dlg.get_settings()
            # Apply language
            if settings['language'] == "日本語":
                self.change_language("ja")
            else:
                self.change_language("en")
            # Apply theme
            self.change_theme(settings['theme'].lower())

    def force_table_refresh(self):
        """Force a complete refresh of all tables."""
        if self.service:
            if self.is_date_specific_view:
                # Refresh with date-specific view
                self.load_events_for_specific_date(self.current_date)
            else:
                # Refresh with regular view
                self.load_events()
    
    def show_snackbar(self, message, duration=3000):
        """Show a temporary notification at the bottom of the window."""
        self.snackbar.show_snackbar(message, duration)

# Translation dictionary for all user-facing strings
TRANSLATIONS = {
    'en': {
        'language': 'Language',
        'english': 'English',
        'japanese': 'Japanese',
        'speech_recognition': 'Speech Recognition',
        'auto_detect': 'Auto-detect',
        'add_event': 'Add Event',
        'logout': 'Logout',
        'login': 'Login',
        'search_by_date': 'Search by Date',
        'theme': 'Theme',
        'light': 'Light',
        'dark': 'Dark',
        'auto_submit': 'Auto-submit after speech',
        'past_events': "Past Events",
        'todays_events': "Today's Events",
        'not_logged_in': 'Not logged in',
        'calendar': 'SEINX Calendar',
        'date_format': '%Y-%m-%d',
        'name': 'Name',
        'location': 'Location',
        'start_date': 'Start Date',
        'end_date': 'End Date',
        'remarks': 'Remarks',
        'all_day_event': 'All Day Event',
        'event_name': 'Event Name:',
        'start_datetime': 'Start Date & Time:',
        'end_datetime': 'End Date & Time:',
        'settings': 'Settings',
        'success': 'Success',
        'event_created': 'Event created successfully!',
        'error': 'Error',
        'event_failed': 'Failed to create event:',
        'delete_event': 'Delete Event',
        'delete_confirm': 'Are you sure you want to delete this event?',
        'yes': 'Yes',
        'no': 'No',
        'event_deleted': 'Event deleted successfully!',
        'event_update_success': 'Event updated successfully!',
        'event_update_failed': 'Failed to update event:',
        'event_update': 'Update Event',
        'remarks_label': 'Remarks:',
        'location_label': 'Location:',
        'ok': 'OK',
        'cancel': 'Cancel',
        'calendar_id': 'Calendar ID:',
        'login_title': 'Login',
        'search_calendar_by_date': 'Search Calendar by Date',
        'select_date': 'Select a date to search:',
        'add_event_title': 'Add Event',
        'update_event_title': 'Update Event',
        'settings_title': 'Settings',
        'language_changed': 'Language changed to English',
        'date_label': 'Date',
        'user_label': 'User',
        'cog_tooltip': 'Settings',
        'upcoming_events': 'Upcoming Events',
        'all_day': 'All day',
        'time': 'Time',
        'date': 'Date',
        'start_time': 'Start Time',
        'end_time': 'End Time',
        'sun': 'Sun',
        'mon': 'Mon',
        'tue': 'Tue',
        'wed': 'Wed',
        'thu': 'Thu',
        'fri': 'Fri',
        'sat': 'Sat',
        'events_for_date': 'Events for {date}',
        'add_disabled_in_search': 'Adding events is disabled in date search mode. Please use "Today" button to return to normal view.',
        'validation_error': 'Validation Error',
        'event_title_required': 'Event title is required.',
        'invalid_start_date': 'Invalid start date.',
        'invalid_end_date': 'Invalid end date.',
        'invalid_start_datetime': 'Invalid start date/time.',
        'invalid_end_datetime': 'Invalid end date/time.',
        'start_date_after_end_date': 'Start date must be before end date.',
        'start_time_after_end_time': 'Start time must be before end time.',
        'start_date_out_of_range': 'Start date must be between 1900 and 10 years from now.',
        'end_date_out_of_range': 'End date must be between 1900 and 10 years from now.',
    },
    'ja': {
        'language': '言語',
        'english': '英語',
        'japanese': '日本語',
        'speech_recognition': '音声認識',
        'auto_detect': '自動検出',
        'add_event': 'イベント追加',
        'logout': 'ログアウト',
        'login': 'ログイン',
        'search_by_date': '日付で検索',
        'theme': 'テーマ',
        'light': 'ライト',
        'dark': 'ダーク',
        'auto_submit': '音声後に自動送信',
        'past_events': "過去のイベント",
        'todays_events': "今日のイベント",
        'not_logged_in': '未接続',
        'calendar': 'SEINXカレンダー',
        'date_format': '%Y/%m/%d',
        'name': '名前',
        'location': '場所',
        'start_date': '開始日',
        'end_date': '終了日',
        'remarks': '備考',
        'all_day_event': '終日イベント',
        'event_name': 'イベント名:',
        'start_datetime': '開始日時:',
        'end_datetime': '終了日時:',
        'settings': '設定',
        'success': '成功',
        'event_created': 'イベントが作成されました！',
        'error': 'エラー',
        'event_failed': 'イベントの作成に失敗しました:',
        'delete_event': 'イベント削除',
        'delete_confirm': 'このイベントを削除してもよろしいですか？',
        'yes': 'はい',
        'no': 'いいえ',
        'event_deleted': 'イベントが削除されました！',
        'event_update_success': 'イベントが更新されました！',
        'event_update_failed': 'イベントの更新に失敗しました:',
        'event_update': 'イベント更新',
        'remarks_label': '備考:',
        'location_label': '場所:',
        'ok': 'OK',
        'cancel': 'キャンセル',
        'calendar_id': 'カレンダーID:',
        'login_title': 'ログイン',
        'search_calendar_by_date': '日付でカレンダー検索',
        'select_date': '検索する日付を選択:',
        'add_event_title': 'イベント追加',
        'update_event_title': 'イベント更新',
        'settings_title': '設定',
        'language_changed': '言語が日本語に変更されました',
        'date_label': '日付',
        'user_label': 'ユーザー',
        'cog_tooltip': '設定',
        'upcoming_events': '予定イベント',
        'all_day': '終日',
        'time': '時間',
        'date': '日付',
        'start_time': '開始時間',
        'end_time': '終了時間',
        'sun': '日',
        'mon': '月',
        'tue': '火',
        'wed': '水',
        'thu': '木',
        'fri': '金',
        'sat': '土',
        'events_for_date': '{date}のイベント',
        'add_disabled_in_search': '日付検索モードではイベント追加が無効です。「今日」ボタンを使用して通常表示に戻してください。',
        'validation_error': '入力エラー',
        'event_title_required': 'イベント名は必須です。',
        'invalid_start_date': '開始日が無効です。',
        'invalid_end_date': '終了日が無効です。',
        'invalid_start_datetime': '開始日時が無効です。',
        'invalid_end_datetime': '終了日時が無効です。',
        'start_date_after_end_date': '開始日は終了日より前である必要があります。',
        'start_time_after_end_time': '開始時刻は終了時刻より前である必要があります。',
        'start_date_out_of_range': '開始日は1900年から10年後までの範囲で入力してください。',
        'end_date_out_of_range': '終了日は1900年から10年後までの範囲で入力してください。',
    }
}

# Central settings object for language and theme
class AppSettings:
    language = 'en'
    theme = 'light'

def tr(key, lang=None):
    if lang is None:
        lang = AppSettings.language
    return TRANSLATIONS.get(lang, TRANSLATIONS['en']).get(key, key)

class Snackbar(QLabel):
    """
    A simple snackbar widget for temporary user notifications.
    Appears at the bottom center of the parent window and fades in/out.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            "background-color: rgba(50, 50, 50, 0.95); color: white; "
            "border-radius: 8px; padding: 12px 32px; font-size: 15px;"
        )
        self.setAlignment(Qt.AlignCenter)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.ToolTip)
        self.setVisible(False)
        self.anim = QPropertyAnimation(self, b"windowOpacity")
        self.anim.setEasingCurve(QEasingCurve.InOutQuad)
        self.anim.finished.connect(self._on_fade_out)
        self._is_showing = False

    def show_snackbar(self, message, duration=3000):
        if self._is_showing:
            self.hide()
        self.setText(message)
        self.adjustSize()
        parent = self.parentWidget()
        if parent:
            x = (parent.width() - self.width()) // 2
            y = parent.height() - self.height() - 40
            self.move(x, y)
        self.setWindowOpacity(0.0)
        self.setVisible(True)
        self.anim.stop()
        self.anim.setDuration(250)
        self.anim.setStartValue(0.0)
        self.anim.setEndValue(1.0)
        self.anim.start()
        QTimer.singleShot(duration, self.fade_out)
        self._is_showing = True

    def fade_out(self):
        self.anim.stop()
        self.anim.setDuration(400)
        self.anim.setStartValue(1.0)
        self.anim.setEndValue(0.0)
        self.anim.start()

    def _on_fade_out(self):
        if self.windowOpacity() == 0.0:
            self.setVisible(False)
            self._is_showing = False

if __name__ == "__main__":
    # --- Startup environment checks ---
    import shutil
    import torch
    from PyQt5.QtWidgets import QApplication, QMessageBox
    import sounddevice as sd
    missing = []
    if shutil.which('ffmpeg') is None:
        missing.append('ffmpeg (required for audio processing)')
    if not torch.cuda.is_available():
        print('Warning: CUDA GPU not detected. Whisper will run on CPU.')
    try:
        devices = sd.query_devices()
        if not any(d['max_input_channels'] > 0 for d in devices):
            missing.append('microphone (no input device found)')
    except Exception:
        missing.append('microphone (error detecting input device)')
    if missing:
        app = QApplication(sys.argv)
        QMessageBox.critical(None, "Missing Dependencies", f"The following are required to run this app:\n- " + "\n- ".join(missing))
        sys.exit(1)
    # --- Normal app startup ---
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())
