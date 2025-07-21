import sys
import json
import os
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

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/calendar']

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
        self.setFixedSize(400, 300)
        
        layout = QVBoxLayout()
        
        # Event name
        layout.addWidget(QLabel("Event Name:"))
        self.name_edit = QLineEdit()
        layout.addWidget(self.name_edit)
        
        # Location
        layout.addWidget(QLabel("Location:"))
        self.location_edit = QLineEdit()
        layout.addWidget(self.location_edit)
        
        # Start date/time
        layout.addWidget(QLabel("Start Date & Time:"))
        self.start_datetime = QDateTimeEdit()
        self.start_datetime.setDateTime(QDateTime.currentDateTime())
        self.start_datetime.setCalendarPopup(True)
        layout.addWidget(self.start_datetime)
        
        # End date/time
        layout.addWidget(QLabel("End Date & Time:"))
        self.end_datetime = QDateTimeEdit()
        self.end_datetime.setDateTime(QDateTime.currentDateTime().addSecs(3600))
        self.end_datetime.setCalendarPopup(True)
        layout.addWidget(self.end_datetime)
        
        # Remarks
        layout.addWidget(QLabel("Remarks:"))
        self.remarks_edit = QTextEdit()
        self.remarks_edit.setMaximumHeight(60)
        layout.addWidget(self.remarks_edit)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.setLayout(layout)
    
    def get_event_data(self):
        return {
            'name': self.name_edit.text(),
            'location': self.location_edit.text(),
            'start': self.start_datetime.dateTime().toPyDateTime(),
            'end': self.end_datetime.dateTime().toPyDateTime(),
            'remarks': self.remarks_edit.toPlainText()
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
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # Set relative column widths
        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Interactive)  # Name
        header.setSectionResizeMode(1, QHeaderView.Interactive)  # Location
        header.setSectionResizeMode(2, QHeaderView.Interactive)  # Start Date
        header.setSectionResizeMode(3, QHeaderView.Interactive)  # End Date
        header.setSectionResizeMode(4, QHeaderView.Stretch)     # Remarks
        
        # Set minimum section sizes
        self.setColumnWidth(0, 200)  # Name
        self.setColumnWidth(1, 150)  # Location
        self.setColumnWidth(2, 160)  # Start Date
        self.setColumnWidth(3, 160)  # End Date
        
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setAlternatingRowColors(True)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)  # Disable editing
        self.cellClicked.connect(self.handle_cell_click)
        
        #empty rows for adding new events
        self.setRowCount(50)
        
        # Create actions widget
        self.actions_widget = None
    
    def handle_cell_click(self, row, column):
        if self.item(row, 0) is None or self.item(row, 0).text() == "":
            # Empty row clicked - add new event
            self.parent_app.add_event()
    
    def show_actions_menu(self, row, event_data):
        menu = QMenu(self)
        update_action = menu.addAction("Update")
        delete_action = menu.addAction("Delete")
        
        update_action.triggered.connect(lambda: self.parent_app.update_event(event_data))
        delete_action.triggered.connect(lambda: self.parent_app.delete_event(event_data))
        
        # Show menu at mouse position
        menu.exec_(QCursor.pos())
    
    def enterEvent(self, event):
        self.setMouseTracking(True)
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        self.setMouseTracking(False)
        super().leaveEvent(event)
        if self.actions_widget:
            self.actions_widget.hide()
    
    def mouseMoveEvent(self, event):
        row = self.rowAt(event.y())
        if row >= 0 and self.item(row, 0) and self.item(row, 0).text():
            self.selectRow(row)
            
            # Show actions for the row
            if self.event_data.get(row):
                # Create container widget for buttons
                if self.actions_widget:
                    self.actions_widget.hide()
                    self.actions_widget.deleteLater()
                
                self.actions_widget = QWidget(self)
                layout = QHBoxLayout(self.actions_widget)
                layout.setSpacing(3)
                layout.setContentsMargins(0, 0, 0, 0)
                
                # Edit button
                edit_btn = QPushButton("Edit", self.actions_widget)
                edit_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #4CAF50;
                        color: white;
                        border: none;
                        padding: 3px 8px;
                        border-radius: 2px;
                        font-size: 11px;
                    }
                    QPushButton:hover {
                        background-color: #45a049;
                    }
                """)
                edit_btn.setCursor(Qt.PointingHandCursor)
                edit_btn.clicked.connect(lambda: self.parent_app.update_event(self.event_data[row]))
                
                # Delete button
                delete_btn = QPushButton("Delete", self.actions_widget)
                delete_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #f44336;
                        color: white;
                        border: none;
                        padding: 3px 8px;
                        border-radius: 2px;
                        font-size: 11px;
                    }
                    QPushButton:hover {
                        background-color: #da190b;
                    }
                """)
                delete_btn.setCursor(Qt.PointingHandCursor)
                delete_btn.clicked.connect(lambda: self.parent_app.delete_event(self.event_data[row]))
                
                layout.addWidget(edit_btn)
                layout.addWidget(delete_btn)
                
                # Position the buttons container in the remarks column
                rect = self.visualItemRect(self.item(row, 4))  # Get remarks column rect
                self.actions_widget.setFixedSize(160, rect.height()-2)
                
                # Calculate position to place buttons at the right side of the remarks cell
                horizontal_pos = rect.x() + rect.width() - 165
                vertical_pos = rect.y() - 1
                
                # Ensure buttons stay within the cell
                if horizontal_pos < rect.x():
                    horizontal_pos = rect.x() + 5  # Add small padding from left if cell is too small
                
                self.actions_widget.move(horizontal_pos, vertical_pos)
                self.actions_widget.show()
        else:
            if self.actions_widget:
                self.actions_widget.hide()
        super().mouseMoveEvent(event)

class UpdateEventDialog(AddEventDialog):
    def __init__(self, event_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Update Event")
        
        # Pre-fill the fields with existing event data
        self.name_edit.setText(event_data.get('summary', ''))
        self.location_edit.setText(event_data.get('location', ''))
        self.remarks_edit.setText(event_data.get('description', ''))
        
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
        self.today_button = QPushButton("Today & Upcoming")
        
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
        lang_menu = menu.addMenu("Language")
        lang_menu.addAction("English", lambda: self.change_language("en"))
        lang_menu.addAction("Spanish", lambda: self.change_language("es"))
        
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
        #language switching logic

        QMessageBox.information(self, "Language", f"Language changed to {lang}")
    
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
                'start': {
                    'dateTime': event_data['start'].isoformat(),
                    'timeZone': 'UTC',
                },
                'end': {
                    'dateTime': event_data['end'].isoformat(),
                    'timeZone': 'UTC',
                },
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
                    'description': updated_data['remarks'],
                    'start': {
                        'dateTime': updated_data['start'].isoformat(),
                        'timeZone': 'UTC',
                    },
                    'end': {
                        'dateTime': updated_data['end'].isoformat(),
                        'timeZone': 'UTC',
                    },
                }
                
                # Preserve any existing fields that we don't update
                for key in event_data:
                    if key not in event and key not in ['start', 'end']:
                        event[key] = event_data[key]
                
                self.service.events().update(
                    calendarId='primary',
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
                    calendarId='primary',
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