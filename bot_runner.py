import os
import time
import threading
import telebot
from flask import Flask, request
import logging

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Config
BOT_TOKEN = os.getenv('BOT_TOKEN')
if not BOT_TOKEN:
    logger.error("❌ BOT_TOKEN not found in environment variables!")
    raise ValueError("BOT_TOKEN is required")

bot = telebot.TeleBot(BOT_TOKEN)
app = Flask(__name__)

# Web server for health check và webhook
@app.route('/')
def home():
    return """
    <h1>🤖 Telegram Bot</h1>
    <p>Bot is running 24/7 on Northflank!</p>
    <p>Owner: KhaMinh7789</p>
    <p><a href="/health">Health Check</a></p>
    """

@app.route('/health')
def health():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "telegram-bot",
        "platform": "northflank"
    }, 200

@app.route('/webhook', methods=['POST'])
def webhook():
    """Endpoint cho webhook (nếu dùng webhook thay vì polling)"""
    if request.headers.get('content-type') == 'application/json':
        json_string = request.get_data().decode('utf-8')
        update = telebot.types.Update.de_json(json_string)
        bot.process_new_updates([update])
        return '', 200
    else:
        return 'Invalid content type', 400

# Bot handlers
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    welcome_text = """
    👋 Xin chào! Tôi là bot của KhaMinh7789

    🤖 **Thông tin bot:**
    - Server: Northflank
    - Status: Always-on 24/7 ✅
    - Owner: @KhaMinh7789

    📝 **Các lệnh có sẵn:**
    /start - Hiển thị thông tin này
    /status - Kiểm tra trạng thái bot
    /info - Thông tin server

    🎯 Bot đang chạy ổn định!
    """
    bot.reply_to(message, welcome_text)

@bot.message_handler(commands=['status'])
def status_command(message):
    status_text = f"""
    🟢 **TRẠNG THÁI BOT**

    ✅ **Hoạt động:** Bình thường
    ⏰ **Uptime:** Đang chạy 24/7
    🌐 **Server:** Northflank
    🐍 **Python:** 3.11
    📊 **Memory:** Optimized

    🎯 Bot ready to serve!
    """
    bot.send_message(message.chat.id, status_text)

@bot.message_handler(commands=['info'])
def info_command(message):
    info_text = f"""
    ℹ️ **THÔNG TIN KỸ THUẬT**

    👨‍💻 **Developer:** KhaMinh7789
    🏢 **Platform:** Northflank
    📦 **Plan:** Free Tier
    🔧 **Type:** Always-on Web Service
    🌍 **Region:** Global

    💡 Bot được deploy tự động từ GitHub
    """
    bot.send_message(message.chat.id, info_text)

@bot.message_handler(func=lambda message: True)
def echo_all(message):
    # Xử lý tin nhắn thông thường
    user_message = message.text
    response = f"🤖 Bot nhận được: '{user_message}'\n\nGõ /help để xem hướng dẫn"
    bot.reply_to(message, response)

def run_web_server():
    """Chạy web server trong thread riêng"""
    try:
        logger.info("🌐 Starting web server on port 8000...")
        app.run(host='0.0.0.0', port=8000, debug=False)
    except Exception as e:
        logger.error(f"❌ Web server error: {e}")

def run_bot():
    """Chạy bot Telegram với auto-restart"""
    logger.info("🤖 Starting Telegram Bot...")
    
    while True:
        try:
            # Dùng polling cho đơn giản
            logger.info("🔄 Bot polling started...")
            bot.polling(none_stop=True, timeout=60, long_polling_timeout=60)
            
        except telebot.apihelper.ApiException as e:
            logger.error(f"❌ Telegram API error: {e}")
            logger.info("🔄 Restarting bot in 30 seconds...")
            time.sleep(30)
            
        except Exception as e:
            logger.error(f"❌ Unexpected bot error: {e}")
            logger.info("🔄 Restarting bot in 60 seconds...")
            time.sleep(60)

def main():
    """Hàm chính khởi chạy mọi thứ"""
    logger.info("🚀 Starting Telegram Bot on Northflank...")
    
    # Validate BOT_TOKEN
    if not BOT_TOKEN:
        logger.error("❌ BOT_TOKEN is not set!")
        return
    
    # Start web server in background thread
    web_thread = threading.Thread(target=run_web_server, daemon=True)
    web_thread.start()
    
    logger.info("✅ Web server started successfully")
    logger.info("📍 Health check available at: http://localhost:8000/health")
    
    # Run bot (main thread)
    run_bot()

if __name__ == "__main__":
    main()