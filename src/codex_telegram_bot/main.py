from __future__ import annotations

import asyncio
import os

import structlog

from .bot import CodexTelegramBot
from .config import Settings
from .logging_utils import configure_logging
from .session_store import SessionStore


logger = structlog.get_logger(__name__)


async def amain() -> None:
    configure_logging(os.getenv("LOG_LEVEL", "INFO"))
    logger.info("app_starting")

    settings = Settings()
    configure_logging(settings.log_level)
    logger.info(
        "settings_loaded",
        approved_directory=str(settings.approved_directory),
        log_level=settings.log_level,
        enable_audit_log=settings.enable_audit_log,
    )

    store = SessionStore(settings.sqlite_path)
    await store.initialize()
    logger.info("session_store_initialized", sqlite_path=str(settings.sqlite_path))

    if await store.health_check():
        logger.info("session_store_healthcheck_ok")
    else:
        logger.error("session_store_healthcheck_failed")
        raise RuntimeError("Session store health check failed")

    bot = CodexTelegramBot(settings, store)
    try:
        bot.codex.validate_cli_available()
    except Exception:
        logger.exception("codex_cli_preflight_failed")
        raise
    logger.info("codex_cli_preflight_ok", codex_cli_path=settings.codex_cli_path)

    logger.info("telegram_app_building")
    app = await bot.build()

    try:
        await app.initialize()
        logger.info("telegram_app_initialized")
        await bot.configure_telegram_ui(app)
        logger.info("telegram_ui_configured")
        await app.start()
        await app.updater.start_polling(drop_pending_updates=True)
        logger.info("polling_started")
        while True:
            await asyncio.sleep(1)
    except Exception:
        logger.exception("app_runtime_failed")
        raise
    finally:
        logger.info("app_shutdown_started")
        try:
            await app.updater.stop()
        except Exception:
            logger.exception("app_shutdown_updater_stop_failed")
        try:
            await app.stop()
        except Exception:
            logger.exception("app_shutdown_app_stop_failed")
        try:
            await app.shutdown()
        except Exception:
            logger.exception("app_shutdown_app_shutdown_failed")
        await store.close()
        logger.info("app_shutdown_completed")


def run() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    run()
