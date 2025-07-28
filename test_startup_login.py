#!/usr/bin/env python3
"""
Test script for startup login functionality
"""

import sys
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QSettings

# Add the current directory to the path so we can import claudever
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_startup_conditions():
    """Test the conditions that determine if login dialog shows on startup"""
    print("=== Testing Startup Login Conditions ===")
    
    # Initialize QApplication
    app = QApplication(sys.argv)
    
    # Check stored calendar ID
    settings = QSettings("SEINX", "Calendar")
    last_calendar_id = settings.value("last_calendar_id", "")
    print(f"1. Stored calendar ID: '{last_calendar_id}'")
    
    # Check token.json
    has_token = os.path.exists('token.json')
    print(f"2. Has token.json: {has_token}")
    
    # Check if user would be logged in
    would_show_dialog = not last_calendar_id or not has_token
    print(f"3. Would show login dialog: {would_show_dialog}")
    
    if would_show_dialog:
        print("   Reason: Missing stored calendar ID or token")
    else:
        print("   Reason: Both calendar ID and token available")
    
    return not would_show_dialog  # Return True if auto-login would work

def test_silent_auto_login():
    """Test if silent auto-login would work"""
    print("\n=== Testing Silent Auto-Login ===")
    
    settings = QSettings("SEINX", "Calendar")
    last_calendar_id = settings.value("last_calendar_id", "")
    has_token = os.path.exists('token.json')
    
    if not has_token or not last_calendar_id:
        print("❌ Cannot test silent auto-login - missing token or calendar ID")
        return False
    
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
        
        creds = Credentials.from_authorized_user_file('token.json', ['https://www.googleapis.com/auth/calendar'])
        
        if creds and creds.valid:
            print("✅ Token is valid")
            
            # Test the connection
            service = build('calendar', 'v3', credentials=creds)
            calendar = service.calendars().get(calendarId=last_calendar_id).execute()
            print(f"✅ Calendar connection successful: {calendar.get('summary', last_calendar_id)}")
            return True
        else:
            print("❌ Token is invalid or expired")
            return False
            
    except Exception as e:
        print(f"❌ Silent auto-login failed: {e}")
        return False

if __name__ == "__main__":
    conditions_ok = test_startup_conditions()
    silent_login_ok = test_silent_auto_login()
    
    print(f"\n=== Summary ===")
    print(f"Auto-login conditions: {'PASS' if conditions_ok else 'FAIL'}")
    print(f"Silent auto-login: {'PASS' if silent_login_ok else 'FAIL'}")
    
    if conditions_ok and silent_login_ok:
        print("🎉 Startup should auto-login silently!")
    elif conditions_ok:
        print("⚠️  Auto-login conditions met but silent login failed")
    else:
        print("📝 Login dialog will show on startup") 