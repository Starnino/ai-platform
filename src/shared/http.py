import httpx
import logging
from fastapi import FastAPI, HTTPException
logger = logging.getLogger(__name__)

async def async_call(client: httpx.AsyncClient, url: str, method: str = "GET", payload: dict | None = None) -> dict:
    try:
        r = await (client.get(url) if method == "GET" else client.post(url, json=payload))
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        logger.error("HTTP %s from %s: %s", e.response.status_code, url, e.response.text)
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        logger.error("Request error to %s: %s", url, e)
        raise HTTPException(status_code=500, detail=f"Error communicating with {url}: {e}")