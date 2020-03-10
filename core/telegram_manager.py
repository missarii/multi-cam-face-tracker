import logging
from telegram import Bot
from telegram.error import TelegramError
from typing import Optional
from pathlib import Path
import asyncio
import time

class TelegramManager:
    def __init__(self, token: str, chat_id: str, rate_limit: int):
        self.token = token
        self.chat_id = chat_id
        self.bot = None
        self.last_sent = 0
        self.min_interval = rate_limit
        self.loop = asyncio.new_event_loop()

    async def _initialize_bot(self):
        """Initialize the Telegram bot asynchronously"""
        try:
            self.bot = Bot(token=self.token)
            await self.bot.get_me()  # Test connection
            logging.info("Telegram bot initialized successfully")
        except TelegramError as e:
            logging.error(f"Failed to initialize Telegram bot: {e}")
            self.bot = None
