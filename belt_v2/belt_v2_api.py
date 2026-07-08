from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import APIStatusError, AuthenticationError, OpenAI

load_dotenv()

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_BALANCE_URL = "https://api.deepseek.com/user/balance"

MODEL = "deepseek-v4-flash"

ROBOT_PROFILE = {
    "name": "BELT",
    "role": "UCSC Campus Robot Assistant",
    "favorite_color": "blue",
    "favorite_animal": "porcupine",
    "favorite_ice_cream_flavor": "mint chocolate chip",
    "rules": [
        "Keep responses short unless the user asks for detail.",
        "Do not claim to see objects, faces, or locations unless a CV module provides that info.",
        "Ask for consent before remembering personal details.",
        "For now, only use text memory. Face recognition is not implemented yet.",
        "If unsure, say you are not sure.",
    ],
    "personality": [
        "friendly",
        "curious",
        "helpful",
        "slightly witty",
        "patient with beginners",
    ],
}


def get_deepseek_api_key() -> str:
    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        raise RuntimeError(
            "Missing DEEPSEEK_API_KEY. Put it in your .env file like:\n"
            "DEEPSEEK_API_KEY=sk-your-key-here"
        )

    return api_key


def make_deepseek_client() -> OpenAI:
    return OpenAI(
        api_key=get_deepseek_api_key(),
        base_url=DEEPSEEK_BASE_URL,
    )


def check_deepseek_balance() -> Dict[str, Any]:
    api_key = get_deepseek_api_key()

    request = urllib.request.Request(
        DEEPSEEK_BALANCE_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        },
        method="GET",
    )

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            raw_data = response.read().decode("utf-8")
            return json.loads(raw_data)

    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")

        if e.code == 401:
            raise RuntimeError(
                "DeepSeek says your API key is invalid. Check DEEPSEEK_API_KEY in your .env file."
            ) from e

        raise RuntimeError(
            f"Could not check DeepSeek balance. HTTP {e.code}: {body}"
        ) from e

    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Could not connect to DeepSeek balance endpoint: {e}"
        ) from e


def print_balance_status() -> bool:
    try:
        balance = check_deepseek_balance()

        is_available = balance.get("is_available", False)
        balance_infos = balance.get("balance_infos", [])

        print("\nDeepSeek balance check:")
        print(f"  API available: {is_available}")

        if balance_infos:
            for info in balance_infos:
                currency = info.get("currency", "UNKNOWN")
                total = info.get("total_balance", "0")
                granted = info.get("granted_balance", "0")
                topped_up = info.get("topped_up_balance", "0")

                print(f"  Currency: {currency}")
                print(f"  Total balance: {total}")
                print(f"  Granted balance: {granted}")
                print(f"  Topped-up balance: {topped_up}")
        else:
            print("  No balance info returned.")

        print()
        return is_available

    except Exception as e:
        print(f"\nBELT: Could not check DeepSeek balance: {e}\n")
        return False


def safe_chat_completion(
    client: OpenAI,
    messages: List[Dict[str, str]],
) -> Optional[str]:
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
        )
        return response.choices[0].message.content

    except AuthenticationError:
        print(
            "BELT: DeepSeek rejected the API key. Check your .env file and make sure "
            "DEEPSEEK_API_KEY is correct."
        )
        return None

    except APIStatusError as e:
        if e.status_code == 402:
            print(
                "BELT: DeepSeek API connection works, but your account has insufficient balance. "
                "Please top up your DeepSeek balance before chatting."
            )
            return None

        if e.status_code == 429:
            print(
                "BELT: DeepSeek says we are sending requests too quickly. "
                "Wait a bit and try again."
            )
            return None

        if e.status_code in [500, 503]:
            print(
                "BELT: DeepSeek's server is having issues right now. "
                "Try again later."
            )
            return None

        print(f"BELT: DeepSeek API error {e.status_code}: {e}")
        return None

    except Exception as e:
        print(f"BELT: Unexpected error: {e}")
        return None