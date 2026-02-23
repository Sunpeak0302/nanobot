"""QQ channel implementation using botpy SDK."""

import asyncio
import inspect
from collections import deque
from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING

from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import QQConfig

try:
    import botpy
    from botpy.message import C2CMessage

    QQ_AVAILABLE = True
except ImportError:
    QQ_AVAILABLE = False
    botpy = None
    C2CMessage = None

if TYPE_CHECKING:
    from botpy.message import C2CMessage


def _make_bot_class(channel: "QQChannel") -> "type[botpy.Client]":
    """Create a botpy Client subclass bound to the given channel."""
    intents = botpy.Intents(public_messages=True, direct_message=True)

    class _Bot(botpy.Client):
        def __init__(self):
            super().__init__(intents=intents)

        async def on_ready(self):
            logger.info("QQ bot ready: {}", self.robot.name)

        async def on_c2c_message_create(self, message: "C2CMessage"):
            await channel._on_message(message)

        async def on_direct_message_create(self, message):
            # nanobot QQ channel currently targets QQ 单聊 (C2C) only.
            logger.debug("Ignored QQ direct_message_create event (non-C2C)")

    return _Bot


class QQChannel(BaseChannel):
    """QQ channel using botpy SDK with WebSocket connection."""

    name = "qq"

    def __init__(self, config: QQConfig, bus: MessageBus):
        super().__init__(config, bus)
        self.config: QQConfig = config
        self._client: "botpy.Client | None" = None
        self._processed_ids: deque = deque(maxlen=1000)
        # QQ C2C de-duplicates by (msg_id, msg_seq). Keep a per-conversation seq.
        self._msg_seq: dict[tuple[str, str], int] = defaultdict(int)

    async def start(self) -> None:
        """Start the QQ bot."""
        if not QQ_AVAILABLE:
            logger.error("QQ SDK not installed. Run: pip install qq-botpy")
            return

        if not self.config.app_id or not self.config.secret:
            logger.error("QQ app_id and secret not configured")
            return

        self._running = True
        BotClass = _make_bot_class(self)
        self._client = BotClass()

        logger.info("QQ bot started (C2C private message)")
        await self._run_bot()

    async def _run_bot(self) -> None:
        """Run the bot connection with auto-reconnect."""
        while self._running:
            try:
                # botpy versions differ: some expose async start(), others sync start().
                if inspect.iscoroutinefunction(self._client.start):
                    await self._client.start(
                        appid=self.config.app_id,
                        secret=self.config.secret,
                    )
                else:
                    await asyncio.to_thread(
                        self._client.start,
                        appid=self.config.app_id,
                        secret=self.config.secret,
                    )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("QQ bot error: {}", e)
            if self._running:
                logger.info("Reconnecting QQ bot in 5 seconds...")
                await asyncio.sleep(5)

    async def stop(self) -> None:
        """Stop the QQ bot."""
        self._running = False

        async def _await_maybe(func: Callable[[], object]) -> None:
            if inspect.iscoroutinefunction(func):
                await func()
            else:
                result = await asyncio.to_thread(func)
                if inspect.isawaitable(result):
                    await result

        if self._client:
            # Prefer explicit close() to ensure aiohttp session closes before loop shutdown.
            close_fn: Callable[[], object] | None = getattr(self._client, "close", None)
            if callable(close_fn):
                try:
                    await _await_maybe(close_fn)
                except Exception as e:
                    logger.debug(f"QQ client close error (ignored): {e}")
            else:
                # Fallback: close underlying BotHttp session if exposed.
                http_obj = getattr(self._client, "http", None)
                http_close: Callable[[], object] | None = getattr(http_obj, "close", None)
                if callable(http_close):
                    try:
                        await _await_maybe(http_close)
                    except Exception as e:
                        logger.debug(f"QQ http close error (ignored): {e}")

            stop_fn: Callable[[], object] | None = getattr(self._client, "stop", None)
            if callable(stop_fn):
                try:
                    await _await_maybe(stop_fn)
                except Exception as e:
                    logger.debug(f"QQ client stop error (ignored): {e}")
        self._client = None
        logger.info("QQ bot stopped")

    async def send(self, msg: OutboundMessage) -> None:
        """Send a message through QQ."""
        if not self._client:
            logger.warning("QQ client not initialized")
            return
        try:
            reply_msg_id = msg.reply_to or str(msg.metadata.get("message_id", "") or "")
            seq_key = (msg.chat_id, reply_msg_id or "__proactive__")
            self._msg_seq[seq_key] += 1
            await self._client.api.post_c2c_message(
                openid=msg.chat_id,
                msg_type=0,
                content=msg.content,
                msg_id=reply_msg_id,
                msg_seq=self._msg_seq[seq_key],
            )
        except Exception as e:
            logger.error("Error sending QQ message: {}", e)

    async def _on_message(self, data: "C2CMessage") -> None:
        """Handle incoming message from QQ."""
        try:
            message_id = str(getattr(data, "id", "") or "")

            # Dedup by message ID
            if message_id:
                if message_id in self._processed_ids:
                    return
                self._processed_ids.append(message_id)

            author = getattr(data, "author", None)
            user_openid = str(getattr(author, "user_openid", "") or "")
            if not user_openid:
                logger.warning("QQ message missing author.user_openid; ignored")
                return

            content = (getattr(data, "content", "") or "").strip()
            if not content:
                return

            logger.info(f"QQ inbound from openid={user_openid}")
            await self._handle_message(
                sender_id=user_openid,
                chat_id=user_openid,
                content=content,
                metadata={
                    "message_id": message_id,
                    "user_openid": user_openid,
                },
            )
        except Exception:
            logger.exception("Error handling QQ message")
