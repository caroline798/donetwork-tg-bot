
import logging
import os
import openai
from langdetect import detect, DetectorFactory
import datetime
import pytz
from collections import defaultdict
import asyncio

# 确保每次检测结果一致
DetectorFactory.seed = 0

from telegram import Update, ChatPermissions
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# --- 配置区域 ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Telegram Bot Tokens 和管理员 ID
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ADMIN_TELEGRAM_ID = int(os.getenv("ADMIN_TELEGRAM_ID"))
REPORT_BOT_TOKEN = os.getenv("REPORT_BOT_TOKEN")

# OpenAI API 配置
openai.api_key = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4.1-mini"

# --- 数据存储 ---
KNOWLEDGE_BASE_PATH = "/home/ubuntu/donetwork_knowledge.txt"
knowledge_base = ""
if os.path.exists(KNOWLEDGE_BASE_PATH):
    with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
        knowledge_base = f.read()
else:
    logger.error(f"知识库文件不存在: {KNOWLEDGE_BASE_PATH}")

# 内存数据存储
daily_stats = {
    "active_users": set(),
    "messages": [],
    "low_confidence_issues": [],
    "group_ids": set()
}
previous_day_member_counts = {}
violation_counts = defaultdict(int) # 存储用户违规次数

# --- AI 核心功能 ---

async def check_violation(message_text: str) -> str | None:
    """使用 OpenAI API 检测消息是否违规"""
    system_prompt = (
        "你是一个内容审核员。请判断以下消息是否包含 FUD（恐惧、不确定、怀疑）、垃圾广告或人身攻击。\n"
        "- FUD: 唱空项目、散布恐慌、无端质疑项目可信度。\n"
        "- SPAM: 推广其他项目、发送无关链接、刷屏。\n"
        "- ATTACK: 骂人、侮辱、威胁他人。\n"
        "如果消息违规，请只返回 'FUD', 'SPAM', 'ATTACK' 中的一个。如果内容正常，请只返回 'NONE'。"
    )
    try:
        response = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message_text}
            ],
            temperature=0,
            max_tokens=10
        )
        result = response.choices[0].message.content.strip().upper()
        if result in ['FUD', 'SPAM', 'ATTACK']:
            return result
        return None
    except Exception as e:
        logger.error(f"AI 内容审核失败: {e}")
        return None

async def get_ai_response(prompt: str, lang: str) -> (str, float):
    system_message_cn = f"你是一个名为 DONetwork Bot 的智能回复机器人，你的任务是在 Telegram 群组中回复用户消息。你需要根据提供的 DONetwork 项目知识库来回答问题，并保持专业不失风趣，活泼但保证项目方的专业形象。如果问题超出知识库范围，或者你认为回答的置信度低于85%，请明确表示你无法直接回答，并建议联系管理员。\n\nDONetwork 项目知识库：\n{knowledge_base}"
    system_message_en = f"You are an intelligent response bot named DONetwork Bot, and your task is to reply to user messages in Telegram groups. You need to answer questions based on the provided DONetwork project knowledge base, maintaining a professional yet humorous style, lively but ensuring the professional image of the project party. If the question is outside the scope of the knowledge base, or if you believe the confidence of the answer is below 85%, please clearly state that you cannot answer directly and suggest contacting an administrator.\n\nDONetwork Project Knowledge Base:\n{knowledge_base}"

    system_message = system_message_cn if lang == 'zh-cn' else system_message_en

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=500,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        
        ai_text = response.choices[0].message.content.strip()
        
        confidence = 0.90
        if any(word in ai_text.lower() for word in ["无法回答", "请联系管理员", "cannot answer", "contact an administrator"]):
            confidence = 0.70
        elif len(ai_text) < 50:
            confidence = 0.80
        
        return ai_text, confidence
        
    except Exception as e:
        logger.error(f"OpenAI API 调用失败: {e}")
        return "抱歉，AI 服务暂时无法响应，请稍后再试。" if lang == 'zh-cn' else "Sorry, the AI service is temporarily unavailable. Please try again later.", 0.5

# AI 总结话题函数
async def summarize_topics(messages: list[str], lang: str) -> str:
    if not messages:
        return "无主要讨论话题。" if lang == 'zh-cn' else "No main discussion topics."
    
    messages_str = '\n'.join(messages)

    if lang == 'zh-cn':
        prompt = f"""请总结以下群组消息中的3-5个主要讨论话题。请用简洁的语言列出。

消息内容：
{messages_str}"""
    else: # lang == 'en'
        prompt = f"""Please summarize 3-5 main discussion topics from the following group messages. List them concisely.

Messages:
{messages_str}"""

    messages_for_summary = [
        {"role": "system", "content": "你是一个善于总结讨论话题的AI助手。" if lang == 'zh-cn' else "You are an AI assistant skilled at summarizing discussion topics."}, 
        {"role": "user", "content": prompt}
    ]

    try:
        response = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages_for_summary,
            temperature=0.5,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"AI 总结话题失败: {e}")
        return "无法总结主要讨论话题。" if lang == 'zh-cn' else "Unable to summarize main discussion topics."

# --- 核心业务逻辑 ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_message = update.message.text
    lang = 'en'
    try:
        if any('\u4e00' <= char <= '\u9fff' for char in user_message):
            lang = 'zh-cn'
        else:
            detected_lang = detect(user_message)
            if detected_lang == 'zh-cn':
                lang = 'zh-cn'
    except Exception as e:
        logger.warning(f"语言检测异常: {e}，默认使用英文")
        lang = 'en'

    greeting_cn = '你好！我是 DONetwork 智能回复机器人，很高兴为您服务。'
    greeting_en = 'Hello! I am DONetwork intelligent reply bot, happy to serve you.'
    await update.message.reply_text(greeting_cn if lang == 'zh-cn' else greeting_en)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not (update.message and update.message.text and (update.message.chat.type in ["group", "supergroup"])):
        return

    user_message = update.message.text
    chat_id = update.message.chat_id
    user = update.message.from_user
    user_id = user.id
    username = user.username or user.full_name

    logger.info(f"收到来自群组 {chat_id} 用户 {username} ({user_id}) 的消息: {user_message}")

    # 1. 内容审核
    violation_type = await check_violation(user_message)
    if violation_type:
        await handle_violation(update, context, user, chat_id, violation_type, user_message)
        return # 违规消息不进行后续处理

    # 2. 收集日活统计
    daily_stats["active_users"].add(user_id)
    daily_stats["messages"].append(user_message)
    daily_stats["group_ids"].add(chat_id)

    # 3. 智能问答
    lang = 'en'
    try:
        if any('\u4e00' <= char <= '\u9fff' for char in user_message):
            lang = 'zh-cn'
        else:
            detected_lang = detect(user_message)
            if detected_lang == 'zh-cn':
                lang = 'zh-cn'
    except Exception as e:
        logger.warning(f"语言检测异常: {e}，默认使用英文")
        lang = 'en'

    ai_response, confidence = await get_ai_response(user_message, lang)

    if confidence < 0.85:
        response_text = "感谢你的问题，我们的管理员会尽快为你解答！" if lang == 'zh-cn' else "Thank you for your question. Our administrators will answer you as soon as possible!"
        await update.message.reply_text(response_text)
        admin_message = f"收到一个需要人工介入的问题：\n\n用户: {username} ({user_id})\n群组ID: {chat_id}\n问题内容: {user_message}\n\nAI 回复置信度: {confidence:.2f}"
        daily_stats["low_confidence_issues"].append({
            "username": username,
            "user_id": user_id,
            "chat_id": chat_id,
            "message": user_message,
            "confidence": confidence
        })
        try:
            # 使用 REPORT_BOT_TOKEN 发送给管理员
            report_app = Application.builder().token(REPORT_BOT_TOKEN).build()
            await report_app.bot.send_message(chat_id=ADMIN_TELEGRAM_ID, text=admin_message)
            logger.info(f"已私聊通知管理员 {ADMIN_TELEGRAM_ID} 关于低置信度问题。")
        except Exception as e:
            logger.error(f"私聊通知管理员失败: {e}")
            await update.message.reply_text("抱歉，通知管理员失败，请稍后再试。" if lang == 'zh-cn' else "Sorry, failed to notify administrator, please try again later.")
    else:
        await update.message.reply_text(ai_response)
        logger.info(f"已在群组 {chat_id} 回复用户 {username} ({user_id})，置信度: {confidence:.2f}")

async def handle_violation(update: Update, context: ContextTypes.DEFAULT_TYPE, user, chat_id: int, violation_type: str, message_text: str):
    """处理违规消息，执行删除、禁言和上报"""
    # 重要提示：机器人必须是群管理员才能执行以下操作：删除消息、禁言用户
    report_bot_app = Application.builder().token(REPORT_BOT_TOKEN).build()
    report_bot = report_bot_app.bot
    user_id = user.id
    username = user.username or user.full_name

    try:
        await update.message.delete()
        logger.info(f"已删除用户 {username} ({user_id}) 的违规消息: {message_text}")
    except Exception as e:
        logger.error(f"删除消息失败 (请检查管理员权限): {e}")
        # 如果删除失败，仍然尝试进行后续惩罚和上报

    violation_counts[user_id] += 1
    count = violation_counts[user_id]
    action_taken = "删除消息并警告"
    mute_duration_text = ""
    mute_seconds = 0

    try:
        if count == 1:
            await context.bot.send_message(chat_id, f"@{username} 您的消息因涉及[{violation_type}]已被删除，请注意言论，共同维护社区环境。")
        elif count == 2:
            mute_seconds = 10 * 60 # 10分钟
            action_taken = "删除消息并禁言10分钟"
            mute_duration_text = " (10分钟)"
        elif count == 3:
            mute_seconds = 24 * 60 * 60 # 1天
            action_taken = "删除消息并禁言1天"
            mute_duration_text = " (1天)"
        else:
            days = count - 2
            mute_seconds = days * 24 * 60 * 60
            action_taken = f"删除消息并禁言{days}天"
            mute_duration_text = f" ({days}天)"
        
        if mute_seconds > 0:
            until_date = datetime.datetime.now(pytz.utc) + datetime.timedelta(seconds=mute_seconds)
            await context.bot.restrict_chat_member(
                chat_id,
                user_id,
                ChatPermissions(can_send_messages=False, can_send_media_messages=False, can_send_polls=False, can_send_other_messages=False, can_add_web_page_previews=False, can_change_info=False, can_invite_users=False, can_pin_messages=False),
                until_date=until_date
            )
            logger.info(f"已禁言用户 {username} ({user_id}) {mute_duration_text}")

    except Exception as e:
        logger.error(f"执行惩罚失败 (请检查管理员权限): {e}")
        action_taken = "删除消息 (惩罚失败)"

    # 上报管理员
    report_text = (f"== 内容审核通知 ==\n"
                   f"用户: {username} ({user_id})\n"
                   f"违规次数: 第 {count} 次\n"
                   f"违规类型: {violation_type}\n"
                   f"违规内容: {message_text}\n"
                   f"处理结果: {action_taken}{mute_duration_text}")
    try:
        await report_bot.send_message(chat_id=ADMIN_TELEGRAM_ID, text=report_text)
        logger.info(f"已上报管理员 {ADMIN_TELEGRAM_ID} 违规事件。")
    except Exception as e:
        logger.error(f"上报管理员失败: {e}")

# --- 定时任务 ---
async def generate_and_send_daily_report(context: ContextTypes.DEFAULT_TYPE):
    global daily_stats, previous_day_member_counts

    logger.info("开始生成每日统计报告...")

    report_bot_app = Application.builder().token(REPORT_BOT_TOKEN).build()
    report_bot = report_bot_app.bot

    # 1. 当天活跃人数
    active_users_count = len(daily_stats["active_users"])

    # 2. 社区总人数变化
    current_member_counts = {}
    total_current_members = 0
    for chat_id in daily_stats["group_ids"]:
        try:
            chat_info = await context.bot.get_chat(chat_id)
            if chat_info.type in ["group", "supergroup"]:
                # Telegram API 获取群成员数量比较复杂，通常需要迭代获取或依赖 bot 自身权限
                # 这里简化处理，假设 chat_info.members_count 可用，或通过其他方式获取
                # 实际应用中可能需要更复杂的逻辑，例如 get_chat_members_count
                count = chat_info.members_count if hasattr(chat_info, 'members_count') else 0 # 占位符
                if count == 0: # 尝试通过 get_chat_member_count 获取
                    try:
                        count = await context.bot.get_chat_member_count(chat_id)
                    except Exception as e:
                        logger.warning(f"无法通过 get_chat_member_count 获取群组 {chat_id} 成员数量: {e}")
                        count = 0

                current_member_counts[chat_id] = count
                total_current_members += count
            else:
                logger.warning(f"Chat ID {chat_id} 不是群组类型，跳过成员统计。")
        except Exception as e:
            logger.warning(f"无法获取群组 {chat_id} 的成员数量: {e}")

    total_previous_members = sum(previous_day_member_counts.values())
    member_change = total_current_members - total_previous_members
    member_change_text = f"总人数变化: {member_change} (昨日总人数: {total_previous_members}, 今日总人数: {total_current_members})"

    # 3. 主要讨论话题
    main_topics = await summarize_topics(daily_stats["messages"], 'zh-cn') # 默认中文报告

    # 4. 用户遇到的问题
    low_confidence_issues_text = "\n".join([
        f"- 用户: {issue['username']} ({issue['user_id']}), 群组: {issue['chat_id']}\n  问题: {issue['message']}\n  AI 置信度: {issue['confidence']:.2f}"
        for issue in daily_stats["low_confidence_issues"]
    ])
    if not low_confidence_issues_text:
        low_confidence_issues_text = "无。"

    report_text = f"""
每日社区统计报告 ({datetime.date.today().strftime("%Y-%m-%d")})

活跃人数: {active_users_count}
{member_change_text}

主要讨论话题:
{main_topics}

需要管理员介入的问题:
{low_confidence_issues_text}
"""

    try:
        await report_bot.send_message(chat_id=ADMIN_TELEGRAM_ID, text=report_text)
        logger.info(f"每日统计报告已发送给管理员 {ADMIN_TELEGRAM_ID}")
    except Exception as e:
        logger.error(f"发送每日统计报告给管理员 {ADMIN_TELEGRAM_ID} 失败: {e}")

    # 重置每日统计数据并更新昨日成员数量
    previous_day_member_counts = current_member_counts
    daily_stats = {
        "active_users": set(),
        "messages": [],
        "low_confidence_issues": [],
        "group_ids": daily_stats["group_ids"].copy() # 保留群组ID
    }
    logger.info("每日统计报告生成并发送完成，数据已重置。")


def main() -> None:
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # 添加定时任务
    job_queue = application.job_queue
    # 每天北京时间晚上8点 (UTC+8) 运行
    tz = pytz.timezone('Asia/Shanghai')
    job_queue.run_daily(generate_and_send_daily_report, time=datetime.time(hour=20, minute=0, tzinfo=tz), name='daily_report')

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()
