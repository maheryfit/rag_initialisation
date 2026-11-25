"""Scrapy spider that scrapes one web page. Run via modules.data_extraction.scrape_web_page."""

import ipaddress
import socket
from datetime import datetime, timezone
from urllib.parse import urlparse

import scrapy
from scrapy.exceptions import IgnoreRequest
from scrapy.http import HtmlResponse


def is_public_url(url: str) -> bool:
    """True only for http(s) URLs whose host resolves exclusively to public IP addresses."""
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            return False
        addresses = socket.getaddrinfo(parsed.hostname, None)
        return all(ipaddress.ip_address(a[4][0].split("%")[0]).is_global for a in addresses)
    except (ValueError, OSError):
        return False


class PublicOnly:
    """Blocks requests to local or private hosts, including redirect targets."""

    def process_request(self, request, spider=None):
        if not is_public_url(request.url):
            raise IgnoreRequest(f"Non-public URL blocked: {request.url}")

    # The downloader resolves DNS again (rebinding), so the connected IP is checked too.
    # ponytail: the request has already been sent by then; only its content is discarded.
    # Pin the resolved IP if this app is ever exposed to untrusted users.
    def process_response(self, request, response, spider=None):
        if response.ip_address is not None and not response.ip_address.is_global:
            raise IgnoreRequest(f"Response from non-public IP blocked: {request.url}")
        return response


class PageSpider(scrapy.Spider):
    name = "page"
    custom_settings = {
        "ROBOTSTXT_OBEY": True,
        "DOWNLOAD_TIMEOUT": 20,
        "DOWNLOAD_MAXSIZE": 5 * 1024 * 1024,
        "RETRY_TIMES": 1,
        "TELNETCONSOLE_ENABLED": False,
        "DOWNLOADER_MIDDLEWARES": {PublicOnly: 50},
    }

    def __init__(self, url, **kwargs):
        super().__init__(**kwargs)
        self.start_urls = [url]

    def parse(self, response):
        if not isinstance(response, HtmlResponse):
            return
        texts = response.xpath(
            "//body//text()[not(ancestor::script or ancestor::style or ancestor::noscript)]"
        ).getall()
        yield {
            "url": response.url,
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "title": response.xpath("string(//title)").get("").strip(),
            "description": response.xpath("string(//meta[@name='description']/@content)").get("").strip(),
            "headings": [
                " ".join(h.xpath("string(.)").get("").split()) for h in response.css("h1, h2, h3")
            ],
            "text": "\n".join(t for t in (" ".join(t.split()) for t in texts) if t),
        }
