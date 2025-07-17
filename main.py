import sys
import json
import os
from datetime import datetime, timedelta
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QStackedWidget,
                             QTableWidget, QTableWidgetItem, QDialog, QFormLayout,
                             QLineEdit, QDateTimeEdit, QTextEdit, QMessageBox)
from PyQt5.QtCore import Qt, QDateTime, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon

# Google Calendar API imports
try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    GOOGLE_APIS_AVAILABLE = True
except ImportError:
    GOOGLE_APIS_AVAILABLE = False
    print("Google APIs not available. Install with: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")

class GoogleCalendarAuth:
    """Handle Google Calendar authentication"""
    
    SCOPES = ['https://www.googleapis.com/auth/calendar']
    
    def __init__(self):
        self.creds = None
        self.service = None
        
    def authenticate(self):
        """Authenticate with Google Calendar API"""
        if not GOOGLE_APIS_AVAILABLE:
            return False
            
        # Token file to store user's access and refresh tokens
        token_file = 'token.json'
        
        if os.path.exists(token_file):
            self.creds = Credentials.from_authorized_user_file(token_file, self.SCOPES)
            
        # If there are no (valid) credentials available, let the user log in
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                # You need to create credentials.json from Google Cloud Console
                if not os.path.exists('credentials.json'):
                    QMessageBox.warning(None, "Missing Credentials", 
                                      "Please add your credentials.json file from Google Cloud Console")
                    return False
                    
                flow = InstalledAppFlow.from_client_secrets_file('credentials.json', self.SCOPES)
                self.creds = flow.run_local_server(port=0)
                
            # Save credentials for next run
            with open(token_file, 'w') as token:
                token.write(self.creds.to_json())
                
        self.service = build('calendar', 'v3', credentials=self.creds)
        return True
        
    def logout(self):
        """Remove stored credentials"""
        if os.path.exists('token.json'):
            os.remove('token.json')
        self.creds = None
        self.service = None

    def get_calendars(self):
        try:
            calendar_list = service.calendarList().list().execute()
            return calendar_list.get("items",[])
        except Httperror as error:
            print(f"Error : {error}")
            return None

        
    def get_events(self, date,calendar_id):
        """Get events for a specific date"""
        if not self.service:
            return []
            
        start_time = datetime.combine(date, datetime.min.time()).isoformat() + 'Z'
        end_time = datetime.combine(date + timedelta(days=1), datetime.min.time()).isoformat() + 'Z'
        
        try:
            events_result = self.service.events().list(
                calendarId=calendar_id,
                timeMin=start_time,
                timeMax=end_time,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            return events
        except Exception as e:
            print(f"Error fetching events: {e}")
            return []
            
    def create_event(self, title, start_datetime, end_datetime, description=""):
        """Create a new calendar event"""
        if not self.service:
            return False
            
        event = {
            'summary': title,
            'description': description,
            'start': {
                'dateTime': start_datetime.isoformat(),
                'timeZone': 'UTC',
            },
            'end': {
                'dateTime': end_datetime.isoformat(),
                'timeZone': 'UTC',
            },
        }
        
        try:
            event = self.service.events().insert(calendarId='primary', body=event).execute()
            return True
        except Exception as e:
            print(f"Error creating event: {e}")
            return False

class CreateEventDialog(QDialog):
    """Dialog for creating new calendar events"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create New Event")
        self.setFixedSize(400, 300)
        self.setModal(True)
        
        self.setup_ui()
        
    def setup_ui(self):
        layout = QFormLayout()
        
        # Event title
        self.title_edit = QLineEdit()
        layout.addRow("Event Title:", self.title_edit)
        
        # Start date/time
        self.start_datetime = QDateTimeEdit()
        self.start_datetime.setDateTime(QDateTime.currentDateTime())
        layout.addRow("Start DateTime:", self.start_datetime)
        
        # End date/time
        self.end_datetime = QDateTimeEdit()
        self.end_datetime.setDateTime(QDateTime.currentDateTime().addSecs(3600))
        layout.addRow("End DateTime:", self.end_datetime)
        
        # Description
        self.description_edit = QTextEdit()
        self.description_edit.setMaximumHeight(100)
        layout.addRow("Description:", self.description_edit)
        
        # Buttons
        button_layout = QHBoxLayout()
        create_btn = QPushButton("Create Event")
        cancel_btn = QPushButton("Cancel")
        
        create_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(create_btn)
        button_layout.addWidget(cancel_btn)
        
        layout.addRow(button_layout)
        self.setLayout(layout)
        
    def get_event_data(self):
        """Return event data from the form"""
        return {
            'title': self.title_edit.text(),
            'start': self.start_datetime.dateTime().toPyDateTime(),
            'end': self.end_datetime.dateTime().toPyDateTime(),
            'description': self.description_edit.toPlainText()
        }

class CalendarView(QWidget):
    """Calendar view showing events for a specific date"""
    
    def __init__(self, auth_manager, parent=None):
        super().__init__(parent)
        self.auth_manager = auth_manager
        self.current_date = datetime.now().date()
        self.parent_window = parent
        
        self.setup_ui()
        self.load_events()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Top bar with back button, date, and create event button
        top_layout = QHBoxLayout()
        
        # Back button
        back_btn = QPushButton("← Back")
        back_btn.clicked.connect(self.go_back)
        top_layout.addWidget(back_btn)
        
        # Date navigation
        date_layout = QHBoxLayout()
        
        self.prev_btn = QPushButton("◀")
        self.prev_btn.clicked.connect(self.previous_date)
        date_layout.addWidget(self.prev_btn)
        
        self.date_label = QLabel()
        self.date_label.setAlignment(Qt.AlignCenter)
        self.date_label.setFont(QFont("Arial", 16, QFont.Bold))
        date_layout.addWidget(self.date_label)
        
        self.next_btn = QPushButton("▶")
        self.next_btn.clicked.connect(self.next_date)
        date_layout.addWidget(self.next_btn)
        
        top_layout.addLayout(date_layout)
        
        # Create event button
        create_btn = QPushButton("Create Event")
        create_btn.clicked.connect(self.create_event)
        top_layout.addWidget(create_btn)
        
        layout.addLayout(top_layout)
        
        # Events table
        self.events_table = QTableWidget()
        self.events_table.setColumnCount(3)
        self.events_table.setHorizontalHeaderLabels(["Time", "Event", "Description"])
        self.events_table.horizontalHeader().setStretchLastSection(True)
        
        layout.addWidget(self.events_table)
        
        self.setLayout(layout)
        self.update_date_label()
        
    def update_date_label(self):
        """Update the date label"""
        self.date_label.setText(self.current_date.strftime("%A, %B %d, %Y"))
        
    def load_events(self):
        """Load events for the current date"""
        events = self.auth_manager.get_events(self.current_date)
        
        self.events_table.setRowCount(len(events))
        
        for i, event in enumerate(events):
            # Get event time
            start = event.get('start', {})
            if 'dateTime' in start:
                start_time = datetime.fromisoformat(start['dateTime'].replace('Z', '+00:00'))
                time_str = start_time.strftime("%H:%M")
            else:
                time_str = "All Day"
                
            # Set table items
            self.events_table.setItem(i, 0, QTableWidgetItem(time_str))
            self.events_table.setItem(i, 1, QTableWidgetItem(event.get('summary', 'No Title')))
            self.events_table.setItem(i, 2, QTableWidgetItem(event.get('description', '')))
            
    def previous_date(self):
        """Navigate to previous date"""
        self.current_date -= timedelta(days=1)
        self.update_date_label()
        self.load_events()
        
    def next_date(self):
        """Navigate to next date"""
        self.current_date += timedelta(days=1)
        self.update_date_label()
        self.load_events()
        
    def create_event(self):
        """Show create event dialog"""
        dialog = CreateEventDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            event_data = dialog.get_event_data()
            
            if event_data['title']:
                success = self.auth_manager.create_event(
                    event_data['title'],
                    event_data['start'],
                    event_data['end'],
                    event_data['description']
                )
                
                if success:
                    QMessageBox.information(self, "Success", "Event created successfully!")
                    self.load_events()
                else:
                    QMessageBox.warning(self, "Error", "Failed to create event.")
            else:
                QMessageBox.warning(self, "Error", "Please enter an event title.")
                
    def go_back(self):
        """Return to dashboard"""
        if self.parent_window:
            self.parent_window.show_dashboard()

class DashboardView(QWidget):
    """Main dashboard view"""
    
    def __init__(self, auth_manager, parent=None):
        super().__init__(parent)
        self.auth_manager = auth_manager
        self.parent_window = parent
        
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Top section with greeting and logout
        top_layout = QHBoxLayout()
        
        # Greeting
        greeting_label = QLabel(f"Good {self.get_time_of_day()}, User!")
        greeting_label.setFont(QFont("Arial", 14, QFont.Bold))
        top_layout.addWidget(greeting_label)
        
        # Logout button
        logout_btn = QPushButton("Logout")
        logout_btn.clicked.connect(self.logout)
        top_layout.addWidget(logout_btn)
        
        layout.addLayout(top_layout)
        
        # Add some spacing
        layout.addStretch()
        
        # View Schedule button
        view_schedule_btn = QPushButton("View Schedule")
        view_schedule_btn.setFont(QFont("Arial", 12))
        view_schedule_btn.clicked.connect(self.view_schedule)
        layout.addWidget(view_schedule_btn)
        
        self.setLayout(layout)
        
    def get_time_of_day(self):
        """Get appropriate greeting based on time"""
        hour = datetime.now().hour
        if hour < 12:
            return "morning"
        elif hour < 18:
            return "afternoon"
        else:
            return "evening"
            
    def logout(self):
        """Logout from Google account"""
        self.auth_manager.logout()
        if self.parent_window:
            self.parent_window.show_login()
            
    def view_schedule(self):
        """Switch to calendar view"""
        if self.parent_window:
            self.parent_window.show_calendar()

class LoginView(QWidget):
    """Login view for Google authentication"""
    
    def __init__(self, auth_manager, parent=None):
        super().__init__(parent)
        self.auth_manager = auth_manager
        self.parent_window = parent
        
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title_label = QLabel("Google Calendar Manager")
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Login button
        login_btn = QPushButton("Login with Google")
        login_btn.setFont(QFont("Arial", 12))
        login_btn.clicked.connect(self.login)
        layout.addWidget(login_btn)
        
        self.setLayout(layout)
        
    def login(self):
        """Handle Google login"""
        if self.auth_manager.authenticate():
            if self.parent_window:
                self.parent_window.show_dashboard()
        else:
            QMessageBox.warning(self, "Login Failed", "Failed to authenticate with Google.")

class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Google Calendar Manager")
        self.setFixedSize(1080, 720)
        
        self.auth_manager = GoogleCalendarAuth()
        
        self.setup_ui()
        
    def setup_ui(self):
        # Central widget with stacked layout
        self.central_widget = QStackedWidget()
        self.setCentralWidget(self.central_widget)
        
        # Create views
        self.login_view = LoginView(self.auth_manager, self)
        self.dashboard_view = DashboardView(self.auth_manager, self)
        self.calendar_view = CalendarView(self.auth_manager, self)
        
        # Add views to stack
        self.central_widget.addWidget(self.login_view)
        self.central_widget.addWidget(self.create_main_layout())
        self.central_widget.addWidget(self.calendar_view)
        
        # Show login view initially
        self.show_login()
        
    def create_main_layout(self):
        """Create the main layout with 30/70 split"""
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # Dashboard (30%)
        dashboard_container = QWidget()
        dashboard_container.setFixedWidth(324)  # 30% of 1080
        dashboard_container.setStyleSheet("background-color: #f0f0f0; border-right: 1px solid #ccc;")
        
        dashboard_layout = QVBoxLayout()
        dashboard_layout.addWidget(self.dashboard_view)
        dashboard_container.setLayout(dashboard_layout)
        
        # Right section (70%)
        right_widget = QWidget()
        right_widget.setFixedWidth(756)  # 70% of 1080
        
        main_layout.addWidget(dashboard_container)
        main_layout.addWidget(right_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        main_widget.setLayout(main_layout)
        return main_widget
        
    def show_login(self):
        """Show login view"""
        self.central_widget.setCurrentWidget(self.login_view)
        
    def show_dashboard(self):
        """Show dashboard view"""
        self.central_widget.setCurrentIndex(1)
        
    def show_calendar(self):
        """Show calendar view"""
        self.central_widget.setCurrentWidget(self.calendar_view)

def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()