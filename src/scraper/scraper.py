"""
Rapid KL tweet scraper for NLP sentiment analysis.

Usage (from project root, venv activated):
    python -m src.scraper.scraper

First-time setup — add at least one Twitter account:
    python -m src.scraper.add_accounts
"""

import asyncio
import csv
import re
from pathlib import Path

from loguru import logger
from twscrape import API

from src.scraper import config


_URL_RE = re.compile(r"https?://\S+")

def has_text(content: str) -> bool:
    return bool(_URL_RE.sub("", content).strip())


def build_query(base: str) -> str:
    q = base
    if config.EXCLUDE_RETWEETS:
        q += " -is:retweet"
    if config.DATE_SINCE:
        q += f" since:{config.DATE_SINCE}"
    if config.DATE_UNTIL:
        q += f" until:{config.DATE_UNTIL}"
    return q


def load_seen_ids(csv_path: Path) -> set[int]:
    seen: set[int] = set()
    if not csv_path.exists():
        return seen
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                seen.add(int(row["id"]))
            except (KeyError, ValueError):
                pass
    return seen


async def scrape_query(
    api: API,
    query: str,
    seen_ids: set[int],
    writer: csv.DictWriter,
) -> int:
    full_query = build_query(query)
    logger.info(f"Searching: {full_query!r}  (limit={config.LIMIT_PER_QUERY})")
    new_count = 0

    async for tweet in api.search(full_query, limit=config.LIMIT_PER_QUERY):
        if tweet.id in seen_ids:
            continue
        if not has_text(tweet.rawContent or ""):
            continue
        seen_ids.add(tweet.id)

        writer.writerow({
            "id":           tweet.id,
            "rawContent":   tweet.rawContent,
            "date":         tweet.date.isoformat() if tweet.date else "",
            "username":     tweet.user.username if tweet.user else "",
            "likeCount":    tweet.likeCount,
            "retweetCount": tweet.retweetCount,
            "replyCount":   tweet.replyCount,
            "lang":         tweet.lang or "",
        })
        new_count += 1

    logger.success(f"  -> {new_count} new tweets from {query!r}")
    return new_count


async def main() -> None:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)

    seen_ids = load_seen_ids(config.OUTPUT_CSV)
    logger.info(f"Loaded {len(seen_ids)} existing tweet IDs for deduplication")

    api = API(str(config.ACCOUNTS_DB))

    # Open CSV in append mode; write header only if the file is new/empty.
    # utf-8-sig adds a BOM on new files so Excel opens them correctly (no garbled emojis).
    # Existing files append with plain utf-8 to avoid inserting a stray BOM mid-file.
    write_header = not config.OUTPUT_CSV.exists() or config.OUTPUT_CSV.stat().st_size == 0
    encoding = "utf-8-sig" if write_header else "utf-8"
    fh = config.OUTPUT_CSV.open("a", encoding=encoding, newline="")
    writer = csv.DictWriter(fh, fieldnames=config.CSV_COLUMNS)
    if write_header:
        writer.writeheader()

    total_new = 0
    try:
        for query in config.SEARCH_QUERIES:
            try:
                count = await scrape_query(api, query, seen_ids, writer)
                total_new += count
                fh.flush()
            except Exception as e:
                logger.error(f"Failed on query {query!r}: {e}")
                continue
    finally:
        fh.close()

    logger.success(f"Done. {total_new} new tweets written to {config.OUTPUT_CSV}")


if __name__ == "__main__":
    asyncio.run(main())