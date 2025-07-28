#!/usr/bin/env python3
"""
Debug script for auto-login functionality
"""

import os
import json
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QSettings

def debug_auto_login_conditions():
    """Debug the auto-login conditions"""
    print("=== Auto-Login Debug ===")
    
    # Initialize QApplication
    app = QApplication([])
    
    # Check token.json
    token_exists = os.path.exists('token.json')
    print(f"1. token.json exists: {token_exists}")
    
    if token_exists:
        try:
            with open('token.json', 'r') as f:
                token_data = json.load(f)
            print(f"   Token data keys: {list(token_data.keys())}")
            print(f"   Has refresh_token: {'refresh_token' in token_data}")
            print(f"   Has expiry: {'expiry' in token_data}")
        except Exception as e:
            print(f"   Error reading token.json: {e}")
    
    # Check stored calendar ID
    settings = QSettings("SEINX", "Calendar")
    last_calendar_id = settings.value("last_calendar_id", "")
    print(f"2. Stored calendar ID: '{last_calendar_id}'")
    
    # Check credentials.json
    credentials_exists = os.path.exists('credentials.json')
    print(f"3. credentials.json exists: {credentials_exists}")
    
    # Overall conditions
    conditions_met = token_exists and last_calendar_id
    print(f"\nAuto-login conditions met: {conditions_met}")
    
    if conditions_met:
        print("✅ Auto-login should work!")
    else:
        print("❌ Auto-login conditions not met:")
        if not token_exists:
            print("   - Missing token.json")
        if not last_calendar_id:
            print("   - No stored calendar ID")
    
    return conditions_met

def test_token_validation():
    """Test if the token is valid"""
    print("\n=== Token Validation Test ===")
    
    if not os.path.exists('token.json'):
        print("No token.json file found")
        return False
    
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        
        creds = Credentials.from_authorized_user_file('token.json', ['https://www.googleapis.com/auth/calendar'])
        
        print(f"Token loaded: {creds is not None}")
        print(f"Token valid: {creds.valid if creds else False}")
        print(f"Token expired: {creds.expired if creds else False}")
        print(f"Has refresh token: {creds.refresh_token is not None if creds else False}")
        
        if creds and creds.expired and creds.refresh_token:
            print("Attempting token refresh...")
            creds.refresh(Request())
            print(f"Token refreshed successfully: {creds.valid}")
        
        return creds and creds.valid
        
    except Exception as e:
        print(f"Error testing token: {e}")
        return False

if __name__ == "__main__":
    conditions_ok = debug_auto_login_conditions()
    token_ok = test_token_validation()
    
    print(f"\n=== Summary ===")
    print(f"Conditions met: {conditions_ok}")
    print(f"Token valid: {token_ok}")
    
    if conditions_ok and token_ok:
        print("🎉 Auto-login should work perfectly!")
    elif conditions_ok:
        print("⚠️  Conditions met but token may be invalid")
    else:
        print("❌ Auto-login will not work") 