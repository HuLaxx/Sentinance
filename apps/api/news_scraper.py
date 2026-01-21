"""
News Scraper Service

Fetches crypto news from multiple sources:
- CoinDesk
- CoinTelegraph
- Decrypt

Uses asyncio for concurrent fetching.
"""

import asyncio
import httpx
from email.utils import parsedate_to_datetime
from datetime import datetime
from typing import Optional
import structlog
from bs4 import BeautifulSoup

log = structlog.get_logger()

# News source configurations
NEWS_SOURCES = {
    "coindesk": {
        "url": "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "type": "rss",
    },
    "cointelegraph": {
        "url": "https://cointelegraph.com/rss",
        "type": "rss",
    },
    "decrypt": {
        "url": "https://decrypt.co/feed",
        "type": "rss",
    },
}


# ============================================
# NEWS ARTICLE MODEL
# ============================================

class NewsArticle:
    """Represents a news article."""
    
    def __init__(
        self,
        title: str,
        url: str,
        source: str,
        published_at: Optional[datetime] = None,
        summary: Optional[str] = None,
        sentiment: Optional[float] = None,
    ):
        self.title = title
        self.url = url
        self.source = source
        self.published_at = published_at or datetime.utcnow()
        self.summary = summary
        self.sentiment = sentiment
    
    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "url": self.url,
            "source": self.source,
            "published_at": self.published_at.isoformat(),
            "summary": self.summary,
            "sentiment": self.sentiment,
        }


# ============================================
# RSS PARSER
# ============================================

def parse_rss(content: str, source: str) -> list[NewsArticle]:
    """Parse RSS feed content into NewsArticle objects."""
    articles = []
    
    try:
        soup = BeautifulSoup(content, "xml")
        items = soup.find_all("item")[:10]  # Get latest 10
        
        for item in items:
            title = item.find("title")
            link = item.find("link")
            description = item.find("description")
            pub_date = item.find("pubDate")

            published_at = None
            if pub_date and pub_date.get_text(strip=True):
                try:
                    published_at = parsedate_to_datetime(pub_date.get_text(strip=True))
                except (TypeError, ValueError):
                    log.warning("rss_pubdate_parse_failed", source=source)
            
            if title and link:
                article = NewsArticle(
                    title=title.get_text(strip=True),
                    url=link.get_text(strip=True),
                    source=source,
                    published_at=published_at,
                    summary=description.get_text(strip=True)[:200] if description else None,
                )
                articles.append(article)
    except Exception as e:
        log.warning("rss_parse_failed", source=source, error=str(e))
    
    return articles


# ============================================
# FETCHERS
# ============================================

async def fetch_rss(client: httpx.AsyncClient, url: str, source: str) -> list[NewsArticle]:
    """Fetch and parse RSS feed."""
    try:
        response = await client.get(url, timeout=10.0)
        if response.status_code == 200:
            articles = parse_rss(response.text, source)
            log.info("rss_fetched", source=source, articles=len(articles))
            return articles
    except Exception as e:
        log.warning("rss_fetch_failed", source=source, error=str(e))
    
    return []


async def fetch_all_news() -> list[NewsArticle]:
    """Fetch news from all sources concurrently."""
    async with httpx.AsyncClient() as client:
        tasks = []
        
        for source_name, config in NEWS_SOURCES.items():
            if config["type"] == "rss":
                tasks.append(fetch_rss(client, config["url"], source_name))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_articles = []
        for result in results:
            if isinstance(result, list):
                all_articles.extend(result)
        
        # Sort by published date (newest first)
        all_articles.sort(key=lambda a: a.published_at, reverse=True)
        
        log.info("news_fetch_complete", total_articles=len(all_articles))
        return all_articles


# ============================================
# SENTIMENT ANALYSIS (BASIC)
# ============================================

POSITIVE_WORDS = {
    "bullish", "surge", "soar", "gain", "rally", "breakout", "moon",
    "adoption", "partnership", "launch", "upgrade", "milestone", "record",
}

NEGATIVE_WORDS = {
    "bearish", "crash", "plunge", "drop", "sell", "dump", "fear",
    "hack", "scam", "rug", "fraud", "warning", "ban", "lawsuit",
}


def analyze_sentiment(text: str) -> float:
    """
    Basic sentiment analysis.
    Returns score from -1 (very negative) to 1 (very positive).
    """
    if not text:
        return 0.0
    
    text_lower = text.lower()
    words = text_lower.split()
    
    positive_count = sum(1 for word in words if word in POSITIVE_WORDS)
    negative_count = sum(1 for word in words if word in NEGATIVE_WORDS)
    
    total = positive_count + negative_count
    if total == 0:
        return 0.0
    
    score = (positive_count - negative_count) / total
    return round(score, 2)


def enrich_with_sentiment(articles: list[NewsArticle]) -> list[NewsArticle]:
    """Add sentiment scores to articles."""
    for article in articles:
        text = f"{article.title} {article.summary or ''}"
        article.sentiment = analyze_sentiment(text)
    return articles


# ============================================
# MAIN FUNCTIONS
# ============================================

async def get_latest_news(limit: int = 20) -> list[dict]:
    """Get latest news with sentiment analysis."""
    articles = await fetch_all_news()
    articles = enrich_with_sentiment(articles)
    return [a.to_dict() for a in articles[:limit]]


async def get_news_by_topic(topic: str, limit: int = 10) -> list[dict]:
    """Get news filtered by topic (BTC, ETH, etc.)."""
    articles = await fetch_all_news()
    
    # Filter by topic in title or summary
    topic_lower = topic.lower()
    filtered = [
        a for a in articles
        if topic_lower in a.title.lower() or 
           (a.summary and topic_lower in a.summary.lower())
    ]
    
    filtered = enrich_with_sentiment(filtered)
    return [a.to_dict() for a in filtered[:limit]]
