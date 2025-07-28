#!/usr/bin/env python3
"""
Test script for refresh token functionality
"""

import sys
import os
import json
from datetime import datetime, timedelta
from PyQt5.QtWidgets import QApplication

# Add the current directory to the path so we can import claudever
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_token_manager():
    """Test the TokenManager functionality"""
    print("=== Testing TokenManager ===")
    
    # Initialize QApplication
    app = QApplication(sys.argv)
    
    # Import after QApplication is initialized
    from claudever import token_manager
    
    # Test loading credentials
    print("1. Testing credential loading...")
    creds = token_manager.load_credentials()
    if creds:
        print(f"   ✅ Credentials loaded successfully")
        print(f"   Token expired: {creds.expired}")
        print(f"   Has refresh token: {creds.refresh_token is not None}")
        print(f"   Token expiry: {creds.expiry}")
    else:
        print("   ❌ No credentials found")
        return False
    
    # Test token validation
    print("\n2. Testing token validation...")
    is_valid = token_manager.is_token_valid()
    print(f"   Token valid: {is_valid}")
    
    # Test token refresh
    print("\n3. Testing token refresh...")
    refreshed_creds = token_manager.refresh_token_if_needed()
    if refreshed_creds:
        print(f"   ✅ Token refresh successful")
        print(f"   New expiry: {refreshed_creds.expiry}")
    else:
        print(f"   ❌ Token refresh failed")
    
    # Test getting valid credentials
    print("\n4. Testing get_valid_credentials...")
    valid_creds = token_manager.get_valid_credentials()
    if valid_creds:
        print(f"   ✅ Valid credentials obtained")
        print(f"   Token expired: {valid_creds.expired}")
    else:
        print(f"   ❌ Could not obtain valid credentials")
    
    return valid_creds is not None

def test_token_file():
    """Test the token.json file structure"""
    print("\n=== Testing Token File ===")
    
    if not os.path.exists('token.json'):
        print("❌ token.json file not found")
        return False
    
    try:
        with open('token.json', 'r') as f:
            token_data = json.load(f)
        
        print("✅ token.json file is valid JSON")
        print(f"   Has token: {'token' in token_data}")
        print(f"   Has refresh_token: {'refresh_token' in token_data}")
        print(f"   Has expiry: {'expiry' in token_data}")
        print(f"   Has scopes: {'scopes' in token_data}")
        
        if 'expiry' in token_data:
            expiry_str = token_data['expiry']
            try:
                expiry_dt = datetime.fromisoformat(expiry_str.replace('Z', '+00:00'))
                now = datetime.now(expiry_dt.tzinfo)
                time_until_expiry = expiry_dt - now
                print(f"   Expiry: {expiry_str}")
                print(f"   Time until expiry: {time_until_expiry}")
                print(f"   Is expired: {time_until_expiry.total_seconds() < 0}")
            except Exception as e:
                print(f"   Error parsing expiry: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading token.json: {e}")
        return False

def test_google_api_connection():
    """Test the Google API connection with current credentials"""
    print("\n=== Testing Google API Connection ===")
    
    try:
        from claudever import token_manager
        from googleapiclient.discovery import build
        
        creds = token_manager.get_valid_credentials()
        if not creds:
            print("❌ No valid credentials available")
            return False
        
        service = build('calendar', 'v3', credentials=creds)
        
        # Test calendar list access
        calendar_list = service.calendarList().list().execute()
        calendars = calendar_list.get('items', [])
        print(f"✅ API connection successful")
        print(f"   Available calendars: {len(calendars)}")
        
        for calendar in calendars[:3]:  # Show first 3 calendars
            print(f"   - {calendar.get('summary', 'Unknown')} ({calendar.get('id', 'Unknown')})")
        
        return True
        
    except Exception as e:
        print(f"❌ API connection failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Refresh Token Functionality")
    print("=" * 40)
    
    token_file_ok = test_token_file()
    token_manager_ok = test_token_manager()
    api_connection_ok = test_google_api_connection()
    
    print(f"\n=== Summary ===")
    print(f"Token file: {'PASS' if token_file_ok else 'FAIL'}")
    print(f"Token manager: {'PASS' if token_manager_ok else 'FAIL'}")
    print(f"API connection: {'PASS' if api_connection_ok else 'FAIL'}")
    
    if token_file_ok and token_manager_ok and api_connection_ok:
        print("🎉 All tests passed! Refresh token functionality is working correctly.")
    else:
        print("❌ Some tests failed. Check the output above for details.") 