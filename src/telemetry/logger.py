import logging
import json
import os
from datetime import datetime
from typing import Any, Dict

class IndustryLogger:
    """
    Structured logger that simulates industry practices.

    - File handler: always logs full JSON (telemetry / failure analysis).
    - Console handler: mirrors logs to the terminal, but its verbosity can be
      tuned via LOG_LEVEL, or silenced entirely with silence_console() so a web
      server's console isn't flooded with error dumps.
    """
    def __init__(self, name: str = "AI-Lab-Agent", log_dir: str = "logs"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.file_handler = None
        self.console_handler = None

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Avoid adding duplicate handlers if the logger is constructed twice.
        if self.logger.handlers:
            self.file_handler = next(
                (h for h in self.logger.handlers if isinstance(h, logging.FileHandler)), None)
            self.console_handler = next(
                (h for h in self.logger.handlers
                 if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)), None)
            return

        # File Handler (JSON) — UTF-8 so Vietnamese content logs correctly.
        log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y-%m-%d')}.log")
        self.file_handler = logging.FileHandler(log_file, encoding="utf-8")
        self.file_handler.setLevel(logging.INFO)

        # Console Handler — verbosity from LOG_LEVEL (default INFO).
        self.console_handler = logging.StreamHandler()
        level_name = os.getenv("LOG_LEVEL", "INFO").upper()
        self.console_handler.setLevel(getattr(logging, level_name, logging.INFO))

        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(self.console_handler)

    def set_console_level(self, level) -> None:
        """Raise/lower how much the console prints (file logging is unaffected)."""
        if self.console_handler is not None:
            self.console_handler.setLevel(level)

    def silence_console(self) -> None:
        """
        Stop printing anything to the console (incl. errors); keep full file logs.
        Used by the web server so error dumps don't flood its terminal.
        """
        if self.console_handler is not None:
            self.console_handler.setLevel(logging.CRITICAL + 1)

    def log_event(self, event_type: str, data: Dict[str, Any]):
        """Logs an event with a timestamp and type."""
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event_type,
            "data": data
        }
        self.logger.info(json.dumps(payload, ensure_ascii=False))

    def info(self, msg: str):
        self.logger.info(msg)

    def error(self, msg: str, exc_info=True):
        self.logger.error(msg, exc_info=exc_info)

# Global logger instance
logger = IndustryLogger()
