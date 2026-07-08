"""
One-time setup: add Twitter/X accounts to the twscrape pool.

Usage (from project root, venv activated):
    python -m src.scraper.add_accounts

Two modes:
  1. Cookie-based (recommended) — paste cookies from browser DevTools, no login needed
  2. Credential-based           — username/password, triggers email verification

How to get cookies (Mode 1):
  1. Open twitter.com in your browser and log in
  2. Open DevTools (F12) -> Application -> Cookies -> https://twitter.com
  3. Copy all cookies as a single string: "key=val; key2=val2; ..."
"""

import asyncio

from loguru import logger
from twscrape import API

from src.scraper.config import ACCOUNTS_DB, DATA_DIR


async def add_account_interactive(api: API) -> None:
    print("\n--- Add a Twitter/X Account ---")
    print("  1. Cookie-based  (recommended — stable, no login step)")
    print("  2. Credential-based  (username/password)")
    mode = input("Choose mode [1/2]: ").strip()

    if mode == "1":
        # Cookie mode — only need username + cookies; other fields are unused by twscrape
        username = input("Twitter username (any identifier): ").strip()
        print("\nGet cookies from: F12 -> Application -> Cookies -> twitter.com")
        cookies = input("Paste cookies (auth_token=...; ct0=...): ").strip()

        # password/email/email_pass are required positional args by twscrape but unused in cookie mode
        await api.pool.add_account(username, "unused", "unused@unused.com", "unused", cookies=cookies)
        logger.success(f"Added @{username} with cookies (no login required)")
    else:
        username   = input("Twitter username: ").strip()
        password   = input("Twitter password: ").strip()
        email      = input("Account email: ").strip()
        email_pass = input("Email password (for IMAP verification, or Enter to skip): ").strip() or ""

        await api.pool.add_account(username, password, email, email_pass)
        logger.info(f"Added @{username}. Logging in (may require email verification)...")
        await api.pool.login_all()
        logger.success(f"Login complete for @{username}")


async def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    api = API(str(ACCOUNTS_DB))

    while True:
        await add_account_interactive(api)
        again = input("\nAdd another account? [y/N]: ").strip().lower()
        if again != "y":
            break

    accounts = await api.pool.get_all()
    print(f"\nPool summary — {len(accounts)} account(s) total:")
    for acc in accounts:
        print(f"  @{acc.username}  active={acc.active}  locks={acc.locks}")


if __name__ == "__main__":
    asyncio.run(main())