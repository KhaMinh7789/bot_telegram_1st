import asyncio
import logging
from datetime import datetime
import numpy as np
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters, CallbackQueryHandler
import aiohttp
import random
import os
from dotenv import load_dotenv

load_dotenv()

# ================== CẤU HÌNH ==================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BOT_TOKEN = os.getenv('BOT_TOKEN')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
BINANCE_API = 'https://api.binance.com/api/v3'
SYMBOL = 'LINKUSDT'

# User data
subscribed_link_users = {}
subscribed_gold_users = set()
chat_histories = {}
user_last_sent = {}  # Sửa lỗi: biến toàn cục cho last_sent

# ================== TÍNH NĂNG MỚI: PRICE ALERTS ==================
price_alerts = {}  # {chat_id: [{"symbol": "LINKUSDT", "target_price": 15.0, "condition": "above/below", "active": True}, ...]}

# ================== TÍNH NĂNG MỚI: PORTFOLIO TRACKING ==================
user_portfolios = {}  # {chat_id: [{"symbol": "LINKUSDT", "amount": 100, "buy_price": 13.5, "current_price": 14.0}, ...]}

# ================== TÍNH NĂNG MỚI: MARKET NEWS ==================
last_news_cache = {}

last_gold_price = None

# ================== TAVILY SEARCH ==================
class TavilySearch:
    @staticmethod
    async def search(query):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    'https://api.tavily.com/search',
                    json={
                        'api_key': TAVILY_API_KEY,
                        'query': query + " (trả lời ngắn gọn bằng tiếng Việt)",
                        'search_depth': 'basic',
                        'include_answer': True,
                        'max_results': 4
                    },
                    timeout=aiohttp.ClientTimeout(total=12)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        answer = data.get('answer', '')
                        results = data.get('results', [])
                        text = []
                        if answer:
                            text.append(f"**Tavily AI:**\n{answer}\n")
                        for r in results[:3]:
                            text.append(f"• {r['title']}\n{r['content'][:150]}...\n🔗 {r['url']}")
                        return "\n".join(text) or "Không tìm thấy thông tin."
                    return "Lỗi Tavily search"
        except:
            return "Tavily tạm thời không hoạt động."

# ================== SIÊU KẾT HỢP: GEMINI + TAVILY ==================
class SuperAI:
    @staticmethod
    async def ask(question, history=None):
        if history is None:
            history = []

        # Bước 1: Luôn lấy dữ liệu mới nhất từ Tavily (không bao giờ die)
        tavily_data = await TavilySearch.search(question)

        # Bước 2: Tạo prompt "bơm" dữ liệu thật cho Gemini
        enhanced_prompt = f"""
Người dùng hỏi: {question}

Thông tin TÌM KIẾM MỚI NHẤT từ Internet (cập nhật real-time):
{tavily_data}

Dựa vào dữ liệu trên, hãy trả lời một cách tự nhiên, thông minh, dễ hiểu bằng tiếng Việt.
Ưu tiên dùng thông tin mới nhất, nếu có mâu thuẫn thì phân tích rõ ràng.
Không cần trích dẫn nguồn trừ khi được hỏi.
        """.strip()

        # Bước 3: Gọi Gemini (với retry tự động như cũ)
        for model in ["gemini-2.5-flash", "gemini-1.5-flash-8b", "gemini-1.5-flash-exp-0827", "gemini-1.5-flash-latest"]:
            for attempt in range(5):
                try:
                    contents = [{"role": "user", "parts": [{"text": enhanced_prompt}]}]
                    if history:
                        for msg in history:
                            role = "user" if msg["role"] == "user" else "model"
                            contents.append({"role": role, "parts": [{"text": msg["content"]}]})

                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
                    async with aiohttp.ClientSession() as session:
                        async with session.post(url, json={"contents": contents}, timeout=30) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                reply = data["candidates"][0]["content"]["parts"][0]["text"]
                                return reply, history + [{"role": "user", "content": question}, {"role": "assistant", "content": reply}]

                            elif resp.status in (429, 503):
                                await asyncio.sleep(2 ** attempt + random.random())
                                continue
                except:
                    await asyncio.sleep(1)
                    continue

        # Nếu Gemini die hoàn toàn → trả luôn Tavily (vẫn ngon!)
        return f"Gemini đang quá tải...\n\nNhưng đây là thông tin mới nhất mình tìm được:\n\n{tavily_data}", history

# ================== BINANCE & GOLD API ==================
class BinanceAPI:
    @staticmethod
    async def get_24h_stats(symbol, max_retries=3):
        for attempt in range(max_retries):
            try:
                timeout = aiohttp.ClientTimeout(total=10)
                async with aiohttp.ClientSession(timeout=timeout) as s:
                    async with s.get(
                        f"{BINANCE_API}/ticker/24hr", 
                        params={'symbol': symbol}
                    ) as r:
                        if r.status == 200:
                            data = await r.json()
                            return data
                        elif r.status == 429:  # Rate limit
                            wait_time = 2 ** attempt
                            logging.warning(f"⏳ Rate limit, chờ {wait_time}s...")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            logging.warning(f"❌ Binance stats status {r.status}, lần thử {attempt + 1}")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(2 ** attempt)
                                continue
            except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as e:
                logging.warning(f"🔌 Lỗi kết nối stats (lần {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
            except Exception as e:
                logging.error(f"❌ Lỗi không xác định trong get_24h_stats: {e}")
                break
        return None

    @staticmethod
    async def get_current_price(symbol, max_retries=3):
        for attempt in range(max_retries):
            try:
                timeout = aiohttp.ClientTimeout(total=10)
                async with aiohttp.ClientSession(timeout=timeout) as s:
                    async with s.get(
                        f"{BINANCE_API}/ticker/price", 
                        params={'symbol': symbol}
                    ) as r:
                        if r.status == 200:
                            data = await r.json()
                            return float(data['price'])
                        elif r.status == 429:
                            wait_time = 2 ** attempt
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            logging.warning(f"❌ Binance price status {r.status}, lần thử {attempt + 1}")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(2 ** attempt)
                                continue
            except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as e:
                logging.warning(f"🔌 Lỗi kết nối price (lần {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
            except Exception as e:
                logging.error(f"❌ Lỗi không xác định trong get_current_price: {e}")
                break
        return None

    async def get_year_klines(symbol=SYMBOL, max_retries=3):
        for attempt in range(max_retries):
            try:
                timeout = aiohttp.ClientTimeout(total=15)
                async with aiohttp.ClientSession(timeout=timeout) as s:
                    async with s.get(
                        f"{BINANCE_API}/klines", 
                        params={'symbol': symbol, 'interval': '1d', 'limit': 365}
                    ) as r:
                        if r.status == 200:
                            data = await r.json()
                            logging.info(f"✅ Lấy dữ liệu klines thành công, số cây nến: {len(data)}")
                            return data
                        else:
                            logging.warning(f"❌ Binance trả về status {r.status}, lần thử {attempt + 1}")
                            if attempt < max_retries - 1:
                                wait_time = 2 ** attempt
                                await asyncio.sleep(wait_time)
                                continue
            except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as e:
                logging.warning(f"🔌 Lỗi kết nối Binance (lần {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    continue
            except Exception as e:
                logging.error(f"❌ Lỗi không xác định trong get_year_klines: {e}")
                break
        
        logging.error("❌ Không thể lấy dữ liệu klines sau tất cả lần thử")
        return None

class GoldPriceAPI:
    @staticmethod
    async def get_gold_price_vn():
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get("https://api.webgia.com/vang-sjc", headers={"User-Agent": "Mozilla/5.0"}) as r:
                    if r.status == 200:
                        data = await r.json()
                        item = data[0]
                        return {
                            'buy': item.get('buy', '0').replace('.', ''),
                            'sell': item.get('sell', '0').replace('.', ''),
                            'time': item.get('updated_at', 'Mới nhất')
                        }
        except: pass
        return {'buy': '14870000', 'sell': '15070000', 'time': 'Fallback'}

async def get_gold_message():
    data = await GoldPriceAPI.get_gold_price_vn()
    fmt = lambda x: f"{int(x)//100000/10:.1f}".replace('.', ',') if x.isdigit() else "N/A"
    return f"""
GIÁ VÀNG SJC VIỆT NAM

SJC (toàn quốc)
   Mua vào:  <b>{fmt(data['buy'])} triệu/lượng</b>
   Bán ra:   <b>{fmt(data['sell'])} triệu/lượng</b>

Cập nhật: {data['time']}
{datetime.now().strftime('%H:%M • %d/%m/%Y')}
    """.strip()

# ================== DỰ BÁO LINK CHI TIẾT ==================
async def analyze_link(symbol=SYMBOL):
    # Thông báo đang xử lý
    logging.info(f"🔍 Bắt đầu phân tích {symbol}...")
    
    # Lấy dữ liệu klines
    klines = await BinanceAPI.get_year_klines(symbol)
    if not klines:
        error_msg = "❌ Không thể kết nối đến Binance để lấy dữ liệu lịch sử. Vui lòng thử lại sau!"
        logging.error(error_msg)
        return error_msg
    
    # Lấy thống kê 24h
    stats = await BinanceAPI.get_24h_stats(symbol)
    if not stats:
        error_msg = "❌ Không thể lấy dữ liệu thống kê 24h từ Binance."
        logging.error(error_msg)
        return error_msg
    
    try:
        # Xử lý dữ liệu
        price = float(stats['lastPrice'])
        change24 = float(stats['priceChangePercent'])
        closes = [float(c[4]) for c in klines]
        
        logging.info(f"✅ Dữ liệu nhận được: giá ${price}, change {change24}%, {len(closes)} ngày")
        
        # Tính RSI
        def rsi(prices):
            if len(prices) < 15:
                return None
            d = np.diff(prices[-15:])
            g, l = np.where(d>0, d, 0), np.where(d<0, -d, 0)
            avg_gain = np.mean(g)
            avg_loss = np.mean(l)
            if avg_loss == 0:
                return 100
            rs = avg_gain / avg_loss
            return round(100 - 100/(1 + rs), 2)

        current_rsi = rsi(closes)
        
        # Phân tích pattern (giữ nguyên logic cũ)
        similar = []
        for i in range(len(closes)-14):
            past_p = closes[i]
            past_r = rsi(closes[:i+15])
            if past_r is None: 
                continue
            if abs(price - past_p)/past_p*100 <= 2.0 and abs(current_rsi - past_r) <= 6:
                similar.append((closes[i+7] - past_p)/past_p*100)

        # ... (phần còn lại của hàm giữ nguyên)
        
        total = len(similar)
        if total == 0:
            return f"""
*LINK/USDT – KHÔNG TÌM THẤY PATTERN TƯƠNG TỰ*

Giá hiện tại: `${price:,.4f}`
RSI (14 ngày): `{current_rsi}`
Trong 365 ngày qua không có tình huống nào giống hiện tại
→ Không thể dự báo 7 ngày tới

{datetime.now().strftime('%H:%M • %d/%m/%Y')}
            """.strip()

        # ... (phần tính toán và kết luận giữ nguyên)

    except KeyError as e:
        error_msg = f"❌ Lỗi dữ liệu từ Binance: thiếu key {e}"
        logging.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"❌ Lỗi xử lý dữ liệu: {str(e)}"
        logging.error(error_msg)
        return error_msg

# ================== GỬI BÁO CÁ NHÂN ==================
async def send_personal_analysis(context: ContextTypes.DEFAULT_TYPE):
    now = datetime.now()
    
    logging.info(f"🔍 Kiểm tra gửi báo cáo. Số user: {len(subscribed_link_users)}")

    for chat_id, interval in list(subscribed_link_users.items()):
        last_sent = user_last_sent.get(chat_id)
        
        time_since_last = (now - last_sent).total_seconds() if last_sent else float('inf')
        
        if last_sent is None or time_since_last >= interval:
            logging.info(f"🟢 Đủ điều kiện gửi cho {chat_id} (interval: {interval}s, time_since_last: {time_since_last:.0f}s)")
            try:
                msg = await analyze_link()
                await context.bot.send_message(chat_id=chat_id, text=msg, parse_mode='HTML')
                user_last_sent[chat_id] = now
                logging.info(f"✅ Đã gửi báo cáo LINK cho {chat_id}")
            except Exception as e:
                logging.error(f"❌ Lỗi khi gửi cho {chat_id}: {str(e)}")
                if "Chat not found" in str(e) or "bot was blocked" in str(e).lower():
                    subscribed_link_users.pop(chat_id, None)
                    if chat_id in user_last_sent:
                        del user_last_sent[chat_id]
                    logging.warning(f"🗑️ Đã xóa {chat_id} do bị chặn/không tồn tại")
        else:
            remaining = interval - time_since_last
            logging.info(f"⏳ Chưa gửi cho {chat_id}, còn {remaining:.0f}s")

async def send_gold_price(context: ContextTypes.DEFAULT_TYPE):
    global last_gold_price
    msg = await get_gold_message()
    if msg == last_gold_price: return
    last_gold_price = msg
    for chat_id in list(subscribed_gold_users):
        try:
            await context.bot.send_message(chat_id=chat_id, text=msg, parse_mode='HTML')
        except:
            subscribed_gold_users.discard(chat_id)

# ================== TÍNH NĂNG MỚI: KIỂM TRA PRICE ALERTS ==================
async def check_price_alerts(context: ContextTypes.DEFAULT_TYPE):
    """Kiểm tra và gửi cảnh báo giá"""
    for chat_id, alerts in list(price_alerts.items()):
        for i, alert in enumerate(alerts[:]):  # Copy để có thể xóa
            if not alert.get('active', True):
                continue
                
            try:
                current_price = await BinanceAPI.get_current_price(alert['symbol'])
                if current_price is None:
                    continue
                    
                target = alert['target_price']
                condition = alert['condition']
                
                triggered = False
                if condition == "above" and current_price >= target:
                    triggered = True
                elif condition == "below" and current_price <= target:
                    triggered = True
                    
                if triggered:
                    # Gửi cảnh báo
                    message = f"🚨 **CẢNH BÁO GIÁ** 🚨\n\n"
                    message += f"💰 {alert['symbol']} đã đạt mục tiêu!\n"
                    message += f"📈 Giá hiện tại: ${current_price:,.4f}\n"
                    message += f"🎯 Điều kiện: {condition.upper()} ${target:,.4f}\n"
                    message += f"⏰ Thời gian: {datetime.now().strftime('%H:%M • %d/%m/%Y')}"
                    
                    await context.bot.send_message(chat_id=chat_id, text=message, parse_mode='HTML')
                    
                    # Vô hiệu hóa cảnh báo sau khi kích hoạt
                    price_alerts[chat_id][i]['active'] = False
                    
            except Exception as e:
                logging.error(f"Lỗi kiểm tra alert {alert['symbol']} cho {chat_id}: {e}")

# ================== TÍNH NĂNG MỚI: MARKET NEWS ==================
async def get_crypto_news():
    """Lấy tin tức crypto mới nhất"""
    try:
        news_data = await TavilySearch.search("tin tức cryptocurrency bitcoin ethereum blockchain mới nhất")
        return news_data
    except Exception as e:
        return f"Không thể lấy tin tức: {str(e)}"

# ================== LỆNH ==================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Hiển thị bàn phím chọn khoảng thời gian"""
    keyboard = [
        [
            InlineKeyboardButton("1 phút", callback_data="60"),
            InlineKeyboardButton("5 phút", callback_data="300"),
            InlineKeyboardButton("10 phút", callback_data="600"),
        ],
        [
            InlineKeyboardButton("30 phút", callback_data="1800"),
            InlineKeyboardButton("1 giờ", callback_data="3600"),
            InlineKeyboardButton("6 giờ", callback_data="21600"),
        ],
        [
            InlineKeyboardButton("12 giờ", callback_data="43200"),
            InlineKeyboardButton("1 ngày", callback_data="86400"),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "🤖 CHỌN KHOẢNG THỜI GIAN NHẬN BÁO CÁO LINK:\n\n"
        "Sau khi chọn, bot sẽ gửi phân tích LINK/USDT tự động theo chu kỳ đã chọn.",
        reply_markup=reply_markup
    )

async def handle_time_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Xử lý khi người dùng chọn thời gian từ inline keyboard"""
    query = update.callback_query
    await query.answer()
    
    chat_id = query.message.chat_id
    interval_seconds = int(query.data)
    
    # Chuyển đổi giây thành text hiển thị
    if interval_seconds < 3600:
        display_text = f"{interval_seconds // 60} phút"
    elif interval_seconds < 86400:
        display_text = f"{interval_seconds // 3600} giờ"
    else:
        display_text = f"{interval_seconds // 86400} ngày"
    
    # Đăng ký user
    subscribed_link_users[chat_id] = interval_seconds
    
    await query.edit_message_text(
        f"✅ ĐÃ ĐĂNG KÝ THÀNH CÔNG!\n\n"
        f"📊 Bạn sẽ nhận phân tích LINK/USDT mỗi: <b>{display_text}</b>\n\n"
        f"📈 Lần phân tích đầu tiên sẽ đến trong 1 phút...\n"
        f"🔍 Dùng /analyze để xem ngay bây giờ\n"
        f"📋 Dùng /mystatus để kiểm tra trạng thái",
        parse_mode='HTML'
    )

async def mystatus(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if chat_id not in subscribed_link_users:
        await update.message.reply_text("❌ Bạn chưa đăng ký nhận thông báo!\n🔔 Dùng /start để đăng ký")
        return
    
    secs = subscribed_link_users[chat_id]
    if secs < 3600:
        txt = f"{secs//60} phút"
    elif secs < 86400:
        txt = f"{secs//3600} giờ"
    else:
        txt = f"{secs//86400} ngày"
    
    await update.message.reply_text(
        f"📊 TRẠNG THÁI HIỆN TẠI:\n\n"
        f"✅ Đang nhận báo cáo LINK mỗi: <b>{txt}</b>\n\n"
        f"🔔 Dùng /start để thay đổi chu kỳ\n"
        f"🚫 Dùng /stop để dừng thông báo",
        parse_mode='HTML'
    )

async def start_gold(update: Update, context: ContextTypes.DEFAULT_TYPE):
    subscribed_gold_users.add(update.effective_chat.id)
    await update.message.reply_text("✅ ĐÃ BẬT BÁO GIÁ VÀNG MỖI 5 PHÚT!")
    await gold_command(update, context)

async def gold_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(await get_gold_message(), parse_mode='HTML')

async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        processing_msg = await update.message.reply_text("⏳ Đang phân tích 365 ngày dữ liệu LINK...")
        result = await analyze_link()
        await processing_msg.edit_text(result)
    except Exception as e:
        logging.error(f"❌ Lỗi trong analyze_command: {e}")
        await update.message.reply_text("❌ Có lỗi xảy ra khi phân tích. Vui lòng thử lại sau!")

async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    subscribed_link_users.pop(chat_id, None)
    subscribed_gold_users.discard(chat_id)
    await update.message.reply_text("✅ Đã hủy tất cả thông báo!")

async def stop_gold(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    subscribed_gold_users.discard(chat_id)
    await update.message.reply_text("✅ Đã tắt báo giá vàng mỗi 5 phút!")

# ================== TÍNH NĂNG MỚI: PRICE ALERTS COMMANDS ==================
async def alert_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Đặt cảnh báo giá: /alert LINKUSDT 15.0 above"""
    if len(context.args) != 3:
        await update.message.reply_text(
            "❌ Sử dụng: /alert <symbol> <price> <above/below>\n"
            "Ví dụ: /alert LINKUSDT 15.0 above\n"
            "Ví dụ: /alert BTCUSDT 50000 below"
        )
        return
    
    symbol = context.args[0].upper()
    try:
        target_price = float(context.args[1])
        condition = context.args[2].lower()
    except ValueError:
        await update.message.reply_text("❌ Giá tiền phải là số!")
        return
    
    if condition not in ['above', 'below']:
        await update.message.reply_text("❌ Điều kiện phải là 'above' hoặc 'below'!")
        return
    
    # Kiểm tra symbol có tồn tại không
    current_price = await BinanceAPI.get_current_price(symbol)
    if current_price is None:
        await update.message.reply_text(f"❌ Không tìm thấy symbol {symbol}!")
        return
    
    chat_id = update.effective_chat.id
    
    # Khởi tạo danh sách alerts nếu chưa có
    if chat_id not in price_alerts:
        price_alerts[chat_id] = []
    
    # Thêm alert mới
    alert_id = len(price_alerts[chat_id]) + 1
    price_alerts[chat_id].append({
        'id': alert_id,
        'symbol': symbol,
        'target_price': target_price,
        'condition': condition,
        'active': True,
        'created_at': datetime.now()
    })
    
    await update.message.reply_text(
        f"✅ ĐÃ ĐẶT CẢNH BÁO!\n\n"
        f"💰 Symbol: {symbol}\n"
        f"🎯 Giá mục tiêu: ${target_price:,.4f}\n"
        f"📊 Điều kiện: {condition.upper()}\n"
        f"💵 Giá hiện tại: ${current_price:,.4f}\n\n"
        f"Dùng /myalerts để xem tất cả cảnh báo"
    )

async def myalerts_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Xem tất cả cảnh báo"""
    chat_id = update.effective_chat.id
    
    if chat_id not in price_alerts or not price_alerts[chat_id]:
        await update.message.reply_text("📭 Bạn chưa có cảnh báo nào!")
        return
    
    message = "📋 **DANH SÁCH CẢNH BÁO CỦA BẠN**\n\n"
    
    for i, alert in enumerate(price_alerts[chat_id], 1):
        status = "🟢 ACTIVE" if alert.get('active', True) else "🔴 INACTIVE"
        message += f"{i}. {alert['symbol']} - ${alert['target_price']:,.4f} {alert['condition'].upper()} - {status}\n"
    
    message += f"\nTổng: {len(price_alerts[chat_id])} cảnh báo"
    
    await update.message.reply_text(message)

async def remove_alert_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Xóa cảnh báo: /remove_alert 1"""
    if not context.args:
        await update.message.reply_text("❌ Sử dụng: /remove_alert <số_thứ_tự>")
        return
    
    try:
        alert_index = int(context.args[0]) - 1
    except ValueError:
        await update.message.reply_text("❌ Số thứ tự phải là số!")
        return
    
    chat_id = update.effective_chat.id
    
    if chat_id not in price_alerts or alert_index < 0 or alert_index >= len(price_alerts[chat_id]):
        await update.message.reply_text("❌ Số thứ tự không hợp lệ!")
        return
    
    removed_alert = price_alerts[chat_id].pop(alert_index)
    await update.message.reply_text(f"✅ Đã xóa cảnh báo: {removed_alert['symbol']} ${removed_alert['target_price']:,.4f}")

# ================== TÍNH NĂNG MỚI: PORTFOLIO COMMANDS ==================
async def add_position_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Thêm vị thế: /add_position LINKUSDT 13.5 100"""
    if len(context.args) != 3:
        await update.message.reply_text(
            "❌ Sử dụng: /add_position <symbol> <giá_mua> <số_lượng>\n"
            "Ví dụ: /add_position LINKUSDT 13.5 100"
        )
        return
    
    symbol = context.args[0].upper()
    try:
        buy_price = float(context.args[1])
        amount = float(context.args[2])
    except ValueError:
        await update.message.reply_text("❌ Giá và số lượng phải là số!")
        return
    
    # Kiểm tra symbol
    current_price = await BinanceAPI.get_current_price(symbol)
    if current_price is None:
        await update.message.reply_text(f"❌ Không tìm thấy symbol {symbol}!")
        return
    
    chat_id = update.effective_chat.id
    
    # Khởi tạo portfolio nếu chưa có
    if chat_id not in user_portfolios:
        user_portfolios[chat_id] = []
    
    # Thêm vị thế mới
    user_portfolios[chat_id].append({
        'symbol': symbol,
        'amount': amount,
        'buy_price': buy_price,
        'current_price': current_price,
        'added_at': datetime.now()
    })
    
    total_value = amount * buy_price
    await update.message.reply_text(
        f"✅ ĐÃ THÊM VỊ THẾ!\n\n"
        f"💰 Symbol: {symbol}\n"
        f"📊 Số lượng: {amount:,}\n"
        f"💵 Giá mua: ${buy_price:,.4f}\n"
        f"💳 Tổng giá trị: ${total_value:,.2f}\n\n"
        f"Dùng /portfolio để xem danh mục"
    )

async def portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Xem danh mục đầu tư"""
    chat_id = update.effective_chat.id
    
    if chat_id not in user_portfolios or not user_portfolios[chat_id]:
        await update.message.reply_text("📭 Danh mục đầu tư trống!")
        return
    
    message = "📊 **DANH MỤC ĐẦU TƯ**\n\n"
    total_portfolio_value = 0
    total_pnl = 0
    
    # Cập nhật giá hiện tại
    for position in user_portfolios[chat_id]:
        current_price = await BinanceAPI.get_current_price(position['symbol'])
        if current_price is not None:
            position['current_price'] = current_price
    
    for i, position in enumerate(user_portfolios[chat_id], 1):
        buy_value = position['amount'] * position['buy_price']
        current_value = position['amount'] * position['current_price']
        pnl = current_value - buy_value
        pnl_percent = (pnl / buy_value) * 100
        
        total_portfolio_value += current_value
        total_pnl += pnl
        
        pnl_emoji = "🟢" if pnl >= 0 else "🔴"
        
        message += f"{i}. **{position['symbol']}**\n"
        message += f"   Số lượng: {position['amount']:,}\n"
        message += f"   Giá mua: ${position['buy_price']:,.4f}\n"
        message += f"   Giá hiện tại: ${position['current_price']:,.4f}\n"
        message += f"   P&L: {pnl_emoji} ${pnl:+.2f} ({pnl_percent:+.2f}%)\n\n"
    
    message += f"**TỔNG DANH MỤC:**\n"
    message += f"💰 Tổng giá trị: ${total_portfolio_value:,.2f}\n"
    message += f"📈 Tổng P&L: ${total_pnl:+.2f}\n"
    
    await update.message.reply_text(message)

async def remove_position_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Xóa vị thế: /remove_position 1"""
    if not context.args:
        await update.message.reply_text("❌ Sử dụng: /remove_position <số_thứ_tự>")
        return
    
    try:
        position_index = int(context.args[0]) - 1
    except ValueError:
        await update.message.reply_text("❌ Số thứ tự phải là số!")
        return
    
    chat_id = update.effective_chat.id
    
    if chat_id not in user_portfolios or position_index < 0 or position_index >= len(user_portfolios[chat_id]):
        await update.message.reply_text("❌ Số thứ tự không hợp lệ!")
        return
    
    removed_position = user_portfolios[chat_id].pop(position_index)
    await update.message.reply_text(f"✅ Đã xóa vị thế: {removed_position['symbol']}")

# ================== TÍNH NĂNG MỚI: MARKET NEWS ==================
async def news_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Tin tức crypto mới nhất"""
    await update.message.reply_text("📰 Đang lấy tin tức mới nhất...")
    
    news = await get_crypto_news()
    await update.message.reply_text(f"📊 **TIN TỨC CRYPTO MỚI NHẤT**\n\n{news}")

# ================== TÍNH NĂNG MỚI: PRICE PROBABILITY PREDICTION ==================
async def calculate_price_probability(symbol, target_price):
    """
    Tính xác suất giá chạm mục tiêu dựa trên dữ liệu lịch sử và chỉ số kỹ thuật
    """
    try:
        # Lấy dữ liệu 1 năm
        klines = await get_year_klines(symbol)
        if not klines:
            return None, "Không lấy được dữ liệu từ Binance!"
        
        closes = [float(c[4]) for c in klines]  # Giá đóng cửa
        highs = [float(c[2]) for c in klines]   # Giá cao nhất
        lows = [float(c[3]) for c in klines]    # Giá thấp nhất
        
        stats = await BinanceAPI.get_24h_stats(symbol)
        current_price = float(stats['lastPrice'])
        
        # Tính các chỉ số kỹ thuật
        def calculate_rsi(prices, period=14):
            if len(prices) < period + 1:
                return None
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gains = np.array([np.mean(gains[i:i+period]) for i in range(len(gains)-period+1)])
            avg_losses = np.array([np.mean(losses[i:i+period]) for i in range(len(losses)-period+1)])
            
            rs = avg_gains / (avg_losses + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        def calculate_volatility(prices, period=20):
            returns = np.diff(np.log(prices))
            return np.std(returns[-period:]) * np.sqrt(365) * 100  # Volatility hàng năm %
        
        # Chỉ số hiện tại
        current_rsi = calculate_rsi(closes)[-1] if calculate_rsi(closes) is not None else 50
        current_volatility = calculate_volatility(closes)
        
        # Phân tích pattern trong quá khứ
        similar_patterns = []
        direction = "above" if target_price > current_price else "below"
        
        # Điều kiện tìm pattern - mở rộng hơn
        search_conditions = [
            (15, 10),   # Điều kiện gốc: RSI chênh 15, giá chênh 10%
            (25, 20),   # Mở rộng lần 1
            (35, 30)    # Mở rộng lần 2
        ]
        
        for rsi_threshold, price_threshold in search_conditions:
            similar_patterns = []  # Reset cho mỗi điều kiện
            
            for i in range(30, len(closes) - 7):  # Bỏ qua 30 ngày đầu và 7 ngày cuối
                past_price = closes[i]
                past_rsi = calculate_rsi(closes[:i+1])
                if past_rsi is None or len(past_rsi) == 0:
                    continue
                past_rsi = past_rsi[-1]
                
                # Điều kiện tìm pattern tương tự: RSI và giá gần nhau
                rsi_diff = abs(past_rsi - current_rsi)
                price_diff_pct = abs(past_price - current_price) / current_price * 100
                
                if rsi_diff <= rsi_threshold and price_diff_pct <= price_threshold:
                    # Kiểm tra 7 ngày tiếp theo có chạm target không
                    future_closes = closes[i+1:i+8]
                    future_highs = highs[i+1:i+8]   # Giá cao nhất trong 7 ngày tới
                    future_lows = lows[i+1:i+8]     # Giá thấp nhất trong 7 ngày tới
                    
                    if direction == "above":
                        # Chạm mục tiêu nếu: giá đóng cửa >= target HOẶC giá cao nhất >= target
                        hit_target = any(price >= target_price for price in future_closes) or \
                                   any(high >= target_price for high in future_highs)
                    else:
                        # Chạm mục tiêu nếu: giá đóng cửa <= target HOẶC giá thấp nhất <= target
                        hit_target = any(price <= target_price for price in future_closes) or \
                                   any(low <= target_price for low in future_lows)
                    
                    similar_patterns.append({
                        'past_price': past_price,
                        'past_rsi': past_rsi,
                        'hit_target': hit_target,
                        'max_future_price': max(future_highs) if direction == "above" else min(future_lows),
                        'condition_level': f"RSI±{rsi_threshold}, Price±{price_threshold}%"
                    })
            
            # Nếu tìm thấy đủ pattern thì dừng
            if len(similar_patterns) >= 5:  # Ít nhất 5 pattern
                break
        
        if not similar_patterns:
            return None, "Không tìm thấy pattern tương tự trong lịch sử ngay cả với điều kiện mở rộng!"
        
        # Tính xác suất
        hit_count = sum(1 for pattern in similar_patterns if pattern['hit_target'])
        total_patterns = len(similar_patterns)
        probability = (hit_count / total_patterns) * 100
        
        # Phân tích thêm
        successful_patterns = [p for p in similar_patterns if p['hit_target']]
        failed_patterns = [p for p in similar_patterns if not p['hit_target']]
        
        avg_rsi_success = np.mean([p['past_rsi'] for p in successful_patterns]) if successful_patterns else 0
        avg_rsi_fail = np.mean([p['past_rsi'] for p in failed_patterns]) if failed_patterns else 0
        
        # Tìm điều kiện tìm kiếm được sử dụng
        used_condition = similar_patterns[0]['condition_level'] if similar_patterns else "N/A"
        
        return {
            'probability': probability,
            'total_patterns': total_patterns,
            'hit_count': hit_count,
            'current_price': current_price,
            'current_rsi': current_rsi,
            'current_volatility': current_volatility,
            'direction': direction,
            'avg_rsi_success': avg_rsi_success,
            'avg_rsi_fail': avg_rsi_fail,
            'price_gap_pct': abs(target_price - current_price) / current_price * 100,
            'search_condition_used': used_condition,
            'successful_examples': [p['max_future_price'] for p in successful_patterns[:3]]  # Ví dụ thành công
        }, None
        
    except Exception as e:
        return None, f"Lỗi tính toán: {str(e)}"

async def probability_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Dự đoán xác suất giá chạm mục tiêu: /probability [symbol] <target_price>"""
    args = context.args
    
    if len(args) < 1:
        await update.message.reply_text(
            "🎯 **DỰ ĐOÁN XÁC SUẤT GIÁ**\n\n"
            "❌ Sử dụng: /probability <giá_mục_tiêu>\n"
            "❌ Hoặc: /probability <symbol> <giá_mục_tiêu>\n\n"
            "💡 Ví dụ:\n"
            "• /probability 15.0 → LINK chạm $15.0\n"
            "• /probability BTCUSDT 50000 → BTC chạm $50,000\n"
            "• /probability ETHUSDT 3000 → ETH chạm $3,000"
        )
        return
    
    # Parse arguments
    if len(args) == 1:
        symbol = "LINKUSDT"
        try:
            target_price = float(args[0])
        except ValueError:
            await update.message.reply_text("❌ Giá mục tiêu phải là số!")
            return
    else:
        symbol = args[0].upper()
        try:
            target_price = float(args[1])
        except ValueError:
            await update.message.reply_text("❌ Giá mục tiêu phải là số!")
            return
    
    # Kiểm tra symbol
    current_price_data = await BinanceAPI.get_current_price(symbol)
    if current_price_data is None:
        await update.message.reply_text(f"❌ Không tìm thấy symbol {symbol} trên Binance!")
        return
    
    processing_msg = await update.message.reply_text(
        f"🔮 Đang phân tích xác suất {symbol} chạm ${target_price:,.2f}..."
    )
    
    # Tính toán xác suất
    result, error = await calculate_price_probability(symbol, target_price)
    
    if error:
        # Hiển thị thông tin debug trong lỗi
        await processing_msg.edit_text(f"❌ {error}")
        return
    
    # Phân loại xác suất
    prob = result['probability']
    if prob >= 80:
        confidence = "RẤT CAO 🟢"
        emoji = "🎯"
    elif prob >= 60:
        confidence = "CAO 🟡" 
        emoji = "📈"
    elif prob >= 40:
        confidence = "TRUNG BÌNH 🟠"
        emoji = "📊"
    elif prob >= 20:
        confidence = "THẤP 🔴"
        emoji = "📉"
    else:
        confidence = "RẤT THẤP 💀"
        emoji = "⚰️"
    
    # Tạo message
    direction_text = "LÊN" if result['direction'] == "above" else "XUỐNG"
    gap_text = f"{result['price_gap_pct']:.1f}%"
    
    message = f"""
🔮 **DỰ ĐOÁN XÁC SUẤT GIÁ** {emoji}

💰 **Symbol:** {symbol}
🎯 **Mục tiêu:** ${target_price:,.4f}
💵 **Giá hiện tại:** ${result['current_price']:,.4f}
📊 **Hướng:** {direction_text} ({gap_text})

📈 **CHỈ SỐ HIỆN TẠI:**
   • RSI (14): {result['current_rsi']:.1f}
   • Biến động: {result['current_volatility']:.1f}%

🎲 **PHÂN TÍCH LỊCH SỬ:**
   • Tìm thấy {result['total_patterns']} pattern tương tự
   • Thành công: {result['hit_count']} lần
   • Điều kiện tìm: {result['search_condition_used']}

🎯 **XÁC SUẤT CHẠM MỤC TIÊU:**
   • **{prob:.1f}%** - {confidence}
"""

    if result['successful_examples']:
        examples_text = ', '.join([f'${x:,.0f}' for x in result['successful_examples']])
        message += f"\n💡 **VÍ DỤ THÀNH CÔNG:** {examples_text}"

    message += f"""
    
⚠️ **Lưu ý:** Đây chỉ là dự đoán dựa trên dữ liệu lịch sử, không phải lời khuyên đầu tư!
    """.strip()
    
    await processing_msg.edit_text(message)

# ================== GEMINI CHAT ==================
async def chat_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("❌ Hãy nhập câu hỏi! Ví dụ: /chat Trump hay Harris đang dẫn trước?")
        return

    question = ' '.join(context.args)
    chat_id = update.effective_chat.id
    processing = await update.message.reply_text("⚡ AI siêu tốc đang trả lời...")

    history = chat_histories.get(chat_id, [])[-10:]
    answer, new_history = await SuperAI.ask(question, history)
    
    chat_histories[chat_id] = new_history[-12:]

    await processing.delete()
    await update.message.reply_text(answer)

async def clear_chat_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Xóa lịch sử chat"""
    chat_id = update.effective_chat.id
    if chat_id in chat_histories:
        chat_histories[chat_id] = []
    await update.message.reply_text("✅ Đã xóa lịch sử chat!")

async def test_api_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Test xem Gemini API có hoạt động không"""
    await update.message.reply_text("🔍 Đang test API...")
    response, _ = await SuperAI.ask("Xin chào, hãy trả lời bằng 1 câu ngắn", [])
    await update.message.reply_text(f"Kết quả: {response}")

# Thêm lệnh vào help_command
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("""
🤖 BOT LINK + VÀNG + AI 24/7

📊 LINK/USDT:
/start → chọn thời gian nhận báo cáo
/analyze → xem phân tích ngay
/mystatus → xem trạng thái hiện tại

💰 GIÁ VÀNG:
/gold → giá vàng SJC ngay lập tức
/start_gold → bật báo vàng mỗi 5 phút
/stop_gold → tắt báo vàng

🚨 PRICE ALERTS:
/alert <symbol> <price> <above/below> → đặt cảnh báo
/myalerts → xem tất cả cảnh báo
/remove_alert <số_TT> → xóa cảnh báo

📈 PORTFOLIO TRACKING:
/add_position <symbol> <giá_mua> <số_lượng> → thêm vị thế
/portfolio → xem danh mục đầu tư
/remove_position <số_TT> → xóa vị thế

🔮 PRICE PROBABILITY:
/probability <giá_mục_tiêu> → dự đoán xác suất
/probability <symbol> <giá_mục_tiêu> → cho coin khác

📰 MARKET NEWS:
/news → tin tức crypto mới nhất

🤖 AI (CÓ WEB SEARCH):
/chat <câu_hỏi> → chat với AI thông minh
/clear_chat → xóa lịch sử chat
/test_api → test Gemini API

⚙️ KHÁC:
/stop → hủy tất cả thông báo
/help → xem hướng dẫn này

💡 VÍ DỤ:
/alert LINKUSDT 15.0 above
/add_position BTCUSDT 50000 0.1
/probability 16.5
/probability BTCUSDT 52000
/news
    """)

# ================== MAIN ==================
def main():
    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("mystatus", mystatus))
    app.add_handler(CommandHandler("analyze", analyze_command))
    app.add_handler(CommandHandler("gold", gold_command))
    app.add_handler(CommandHandler("start_gold", start_gold))
    app.add_handler(CommandHandler("stop_gold", stop_gold))
    app.add_handler(CommandHandler("stop", stop))
    
    # Thêm handlers cho tính năng mới
    app.add_handler(CommandHandler("alert", alert_command))
    app.add_handler(CommandHandler("myalerts", myalerts_command))
    app.add_handler(CommandHandler("remove_alert", remove_alert_command))
    app.add_handler(CommandHandler("add_position", add_position_command))
    app.add_handler(CommandHandler("portfolio", portfolio_command))
    app.add_handler(CommandHandler("remove_position", remove_position_command))
    app.add_handler(CommandHandler("news", news_command))
    app.add_handler(CommandHandler("probability", probability_command))
    
    app.add_handler(CommandHandler("chat", chat_command))
    app.add_handler(CommandHandler("clear_chat", clear_chat_command))
    app.add_handler(CommandHandler("test_api", test_api_command))
    app.add_handler(CommandHandler("help", help_command))
    
    # Thêm handler cho inline keyboard
    app.add_handler(CallbackQueryHandler(handle_time_selection, pattern="^(60|300|600|1800|3600|21600|43200|86400)$"))

    jq = app.job_queue
    jq.run_repeating(send_personal_analysis, interval=55, first=10)
    jq.run_repeating(send_gold_price, interval=300, first=15)
    jq.run_repeating(check_price_alerts, interval=30, first=20)  # Kiểm tra alerts mỗi 30 giây

    print("🤖 Bot đã chạy với 3 tính năng mới: Price Alerts, Portfolio Tracking, Market News!")
    app.run_polling()

if __name__ == '__main__':
    main()