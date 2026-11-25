"""Module for scraping a web page and storing the result as JSON."""

import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any
from urllib.parse import urlparse

import config

# Do not import scrapy or web_spider here: loading Scrapy in the same process breaks Gradio.

logger = logging.getLogger(__name__)

SPIDER_PATH = Path(__file__).with_name("web_spider.py")


def output_path(url: str) -> Path:
    """JSON file where the scraped data for ``url`` is stored."""
    host = (urlparse(url).hostname or "page").replace(":", "-")
    digest = hashlib.sha256(url.encode()).hexdigest()[:10]
    return Path(config.SCRAPED_DATA_DIR) / f"{host}-{digest}.json"


def scrape_web_page(url: str) -> Dict[str, Any]:
    """Scrape one public web page with Scrapy and save it to a JSON file.

    Sites whose robots.txt disallows crawling (LinkedIn, for example) return no data.

    Args:
        url: http(s) URL of the page.

    Returns:
        Dictionary with the scraped page data (empty dict if nothing could be scraped).
    """
    url = url.strip()
    path = output_path(url).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)

    # Subprocess because Scrapy's reactor cannot be restarted inside a long-running app.
    command = [
        sys.executable, "-m", "scrapy", "runspider", str(SPIDER_PATH),
        "-a", f"url={url}", "-O", f"{path.as_uri()}:json",
        "-s", "FEED_EXPORT_ENCODING=utf-8",
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90
        )
        items = json.loads(path.read_text(encoding="utf-8"))
        data = items[0]
    except (subprocess.TimeoutExpired, OSError, ValueError, IndexError) as e:
        path.unlink(missing_ok=True)
        logger.error("Scraping failed for %s: %s", url, e)
        if "result" in locals():
            logger.error("Scrapy log (end):\n%s", result.stderr[-2000:])
        return {}

    logger.info("Saved scraped data to %s", path)
    return data
