"""
CLI tool to create the first admin account.

Usage:
    python create_admin.py

This script creates an admin user in the SQLite database.
Run this once after setting up the application.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from getpass import getpass


def main():
    # Import after path setup
    from app.user_db import get_user_by_username, create_user, count_admins, init_db
    from app.auth import get_password_hash
    
    print("\n" + "=" * 50)
    print("🐄 Cattle Estrus Detection - Admin Setup")
    print("=" * 50 + "\n")
    
    # Initialize database
    init_db()
    
    # Check if admin already exists
    admin_count = count_admins()
    if admin_count > 0:
        print(f"ℹ️  There are already {admin_count} admin account(s) in the system.")
        response = input("Do you still want to create another admin? (y/n): ").strip().lower()
        if response != 'y':
            print("Cancelled.")
            return
    
    # Get admin details
    print("\nEnter details for the new admin account:\n")
    
    username = input("Username: ").strip()
    if not username or len(username) < 3:
        print("❌ Username must be at least 3 characters.")
        return
    
    # Check if username exists
    existing = get_user_by_username(username)
    if existing:
        print(f"❌ Username '{username}' already exists.")
        return
    
    password = getpass("Password: ")
    if len(password) < 6:
        print("❌ Password must be at least 6 characters.")
        return
    
    password_confirm = getpass("Confirm Password: ")
    if password != password_confirm:
        print("❌ Passwords do not match.")
        return
    
    full_name = input("Full Name (optional): ").strip() or None
    email = input("Email (optional): ").strip() or None
    
    # Create admin user
    user_data = {
        "username": username,
        "email": email,
        "full_name": full_name,
        "role": "admin",
        "is_active": True,
        "hashed_password": get_password_hash(password)
    }
    
    try:
        new_user = create_user(user_data)
        print("\n" + "=" * 50)
        print("✅ Admin account created successfully!")
        print("=" * 50)
        print(f"\n   Username: {new_user['username']}")
        print(f"   Role: admin")
        print(f"   ID: {new_user['id']}")
        print("\n   You can now login at: http://localhost:8000/login")
        print("=" * 50 + "\n")
    except Exception as e:
        print(f"❌ Failed to create admin: {e}")


if __name__ == "__main__":
    main()
