#!/usr/bin/env python3
"""
File chạy bot trên PythonAnywhere
"""
import sys
import os

# Thêm thư mục hiện tại vào path
sys.path.append(os.path.dirname(__file__))

from bot_tele_coin import main

if __name__ == '__main__':
    print("🤖 Starting Telegram Bot on PythonAnywhere...")
    main()