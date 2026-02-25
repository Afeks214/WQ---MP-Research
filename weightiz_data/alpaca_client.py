from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


LOGGER = logging.getLogger(__name__)
LIMIT_SAFE = 1_000


class AlpacaAPIError(RuntimeError):
    """Base error for Alpaca API failures."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_text: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_text = str(response_text or "")


class AlpacaPermissionError(AlpacaAPIError):
    """Raised when the requested feed or endpoint is not authorized."""


@dataclass(frozen=True)
class AlpacaRequestSpec:
    symbols: Tuple[str, ...]
    timeframe: str
    start: str
    end: str
    feed: str
    adjustment: str
    sort: str = "asc"
    limit: int = 10_000


class AlpacaClient:
    """Thin deterministic client for Alpaca historical stock bars API v2."""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        base_url: str = "https://data.alpaca.markets",
        rate_limit_sleep_sec: float = 0.25,
        timeout_sec: float = 30.0,
        max_retries_429: int = 6,
        backoff_base_sec: float = 0.5,
        backoff_max_sec: float = 8.0,
    ) -> None:
        try:
            import requests  # type: ignore
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError("requests is required. Install with: pip install requests") from exc

        self._requests = requests
        self._base_url = base_url.rstrip("/")
        self._sleep = float(rate_limit_sleep_sec)
        self._timeout = float(timeout_sec)
        self._max_retries_429 = max(0, int(max_retries_429))
        self._backoff_base_sec = max(0.0, float(backoff_base_sec))
        self._backoff_max_sec = max(0.0, float(backoff_max_sec))
        self._session = requests.Session()
        self._session.headers.update(
            {
                "APCA-API-KEY-ID": api_key,
                "APCA-API-SECRET-KEY": secret_key,
                "Accept": "application/json",
            }
        )

    def fetch_bars_multi(
        self,
        symbols: Sequence[str],
        timeframe: str,
        start: str,
        end: str,
        feed: str,
        adjustment: str,
        sort: str = "asc",
        max_symbols_per_request: int = 100,
        limit: int = 10_000,
    ) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, str], List[str]]:
        """
        Fetch multi-symbol historical bars from /v2/stocks/bars with pagination.

        Returns:
        - bars_by_symbol: canonicalized symbol->list[raw_bar_dict]
        - feed_used_by_symbol: symbol->feed actually used (sip/iex)
        - warnings: deterministic warning messages
        """
        bars_by_symbol, feed_used_by_symbol, warnings, _ = self.fetch_bars_multi_with_meta(
            symbols=symbols,
            timeframe=timeframe,
            start=start,
            end=end,
            feed=feed,
            adjustment=adjustment,
            sort=sort,
            max_symbols_per_request=max_symbols_per_request,
            limit=limit,
        )
        return bars_by_symbol, feed_used_by_symbol, warnings

    def fetch_bars_multi_with_meta(
        self,
        symbols: Sequence[str],
        timeframe: str,
        start: str,
        end: str,
        feed: str,
        adjustment: str,
        sort: str = "asc",
        max_symbols_per_request: int = 100,
        limit: int = 10_000,
    ) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, str], List[str], Dict[str, Dict[str, Any]]]:
        """
        Fetch multi-symbol bars and return deterministic per-symbol telemetry metadata.
        """
        syms = [str(s).strip().upper() for s in symbols if str(s).strip()]
        if not syms:
            raise RuntimeError("No symbols provided for Alpaca fetch")

        out: Dict[str, List[Dict[str, Any]]] = {s: [] for s in syms}
        feed_used_by_symbol: Dict[str, str] = {}
        meta_by_symbol: Dict[str, Dict[str, Any]] = {}
        warnings: List[str] = []

        chunk_sz = max(1, int(max_symbols_per_request))
        for i in range(0, len(syms), chunk_sz):
            chunk = tuple(syms[i : i + chunk_sz])
            spec = AlpacaRequestSpec(
                symbols=chunk,
                timeframe=str(timeframe),
                start=str(start),
                end=str(end),
                feed=str(feed).lower(),
                adjustment=str(adjustment).lower(),
                sort=str(sort).lower(),
                limit=int(limit),
            )
            bars_chunk, used_feed, chunk_warnings, chunk_meta = self._fetch_chunk_with_feed_fallback(spec)
            warnings.extend(chunk_warnings)

            for sym in chunk:
                out[sym].extend(bars_chunk.get(sym, []))
                feed_used_by_symbol[sym] = used_feed
                meta = dict(chunk_meta)
                meta["feed_used"] = used_feed
                meta_by_symbol[sym] = meta

        # Deterministic ordering within each symbol.
        for sym in syms:
            out[sym] = sorted(out[sym], key=self._bar_sort_key)

        return out, feed_used_by_symbol, warnings, meta_by_symbol

    def _fetch_chunk_with_feed_fallback(
        self,
        spec: AlpacaRequestSpec,
    ) -> Tuple[Dict[str, List[Dict[str, Any]]], str, List[str], Dict[str, Any]]:
        warnings: List[str] = []
        feed_req = spec.feed

        try:
            data, meta = self._fetch_chunk(spec)
            return data, feed_req, warnings, meta
        except AlpacaPermissionError as exc:
            if feed_req == "sip":
                msg = (
                    "SIP feed not authorized for this key; falling back to IEX. "
                    "IEX has reduced market coverage versus SIP."
                )
                LOGGER.warning(msg)
                warnings.append(msg)
                spec2 = AlpacaRequestSpec(
                    symbols=spec.symbols,
                    timeframe=spec.timeframe,
                    start=spec.start,
                    end=spec.end,
                    feed="iex",
                    adjustment=spec.adjustment,
                    sort=spec.sort,
                    limit=spec.limit,
                )
                data, meta = self._fetch_chunk(spec2)
                return data, "iex", warnings, meta
            raise exc

    def _fetch_chunk(self, spec: AlpacaRequestSpec) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
        endpoint = f"{self._base_url}/v2/stocks/bars"
        out: Dict[str, List[Dict[str, Any]]] = {s: [] for s in spec.symbols}

        limit_requested = int(spec.limit)
        limit_effective = int(spec.limit)
        limit_fallback_used = False
        limit_fallback_reason = ""

        retry_429_count = 0
        backoff_schedule_seconds: List[float] = []

        page_token: Optional[str] = None
        while True:
            params: Dict[str, Any] = {
                "symbols": ",".join(spec.symbols),
                "timeframe": spec.timeframe,
                "start": spec.start,
                "end": spec.end,
                "feed": spec.feed,
                "adjustment": spec.adjustment,
                "sort": spec.sort,
                "limit": int(limit_effective),
            }
            if page_token:
                params["page_token"] = page_token

            try:
                payload, retry_meta = self._request_payload_with_retry(endpoint=endpoint, params=params)
            except AlpacaAPIError as exc:
                status_code = int(getattr(exc, "status_code", 0) or 0)
                err_text = str(getattr(exc, "response_text", "") or str(exc))
                if (
                    not limit_fallback_used
                    and int(limit_effective) != int(LIMIT_SAFE)
                    and self._is_invalid_limit_error(status_code=status_code, text=err_text)
                ):
                    limit_effective = int(LIMIT_SAFE)
                    limit_fallback_used = True
                    limit_fallback_reason = "invalid_limit_400_422"
                    LOGGER.warning(
                        "Alpaca invalid limit rejected; falling back deterministically to limit=%d",
                        int(LIMIT_SAFE),
                    )
                    continue
                raise

            retry_429_count += int(retry_meta["retry_429_count"])
            backoff_schedule_seconds.extend(list(retry_meta["backoff_schedule_seconds"]))

            page = self._extract_bars(payload)
            for sym, bars in page.items():
                if sym not in out:
                    out[sym] = []
                out[sym].extend(bars)

            page_token = payload.get("next_page_token")
            if not page_token:
                break
            if self._sleep > 0:
                time.sleep(self._sleep)

        meta = {
            "limit_requested": int(limit_requested),
            "limit_effective": int(limit_effective),
            "limit_fallback_used": bool(limit_fallback_used),
            "limit_fallback_reason": str(limit_fallback_reason),
            "retry_429_count": int(retry_429_count),
            "backoff_schedule_seconds": [float(x) for x in backoff_schedule_seconds],
            "total_sleep_seconds": float(sum(backoff_schedule_seconds)),
            "sort": str(spec.sort),
            "timeframe": str(spec.timeframe),
            "adjustment": str(spec.adjustment),
        }
        return out, meta

    def _request_payload_with_retry(self, endpoint: str, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        attempt = 0
        retry_429_count = 0
        backoff_schedule_seconds: List[float] = []
        while True:
            resp = self._session.get(endpoint, params=params, timeout=self._timeout)
            if int(resp.status_code) == 429 and attempt < self._max_retries_429:
                sleep_sec = self._retry_backoff_seconds(
                    attempt=attempt,
                    base_sec=self._backoff_base_sec,
                    max_sec=self._backoff_max_sec,
                )
                LOGGER.warning(
                    "Alpaca rate limited (429). retry=%d/%d sleep=%.3fs",
                    attempt + 1,
                    self._max_retries_429,
                    sleep_sec,
                )
                if sleep_sec > 0.0:
                    time.sleep(sleep_sec)
                retry_429_count += 1
                backoff_schedule_seconds.append(float(sleep_sec))
                attempt += 1
                continue

            payload = self._parse_response(resp)
            telemetry = {
                "retry_429_count": int(retry_429_count),
                "backoff_schedule_seconds": [float(x) for x in backoff_schedule_seconds],
            }
            return payload, telemetry

    def _parse_response(self, resp: Any) -> Dict[str, Any]:
        status = int(resp.status_code)
        if status < 200 or status >= 300:
            txt = ""
            try:
                txt = str(resp.text)
            except Exception:
                txt = ""
            msg = f"Alpaca API error status={status}: {txt[:500]}"
            lower = txt.lower()
            if status in (401, 403) or "not authorized" in lower or "forbidden" in lower:
                raise AlpacaPermissionError(msg, status_code=status, response_text=txt)
            raise AlpacaAPIError(msg, status_code=status, response_text=txt)

        try:
            payload = resp.json()
        except Exception as exc:
            raise AlpacaAPIError("Alpaca response is not valid JSON", status_code=status) from exc

        if not isinstance(payload, dict):
            raise AlpacaAPIError(f"Unexpected Alpaca payload type: {type(payload).__name__}", status_code=status)
        return payload

    @staticmethod
    def _extract_bars(payload: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        bars_obj = payload.get("bars")
        if bars_obj is None:
            return {}

        out: Dict[str, List[Dict[str, Any]]] = {}
        if isinstance(bars_obj, dict):
            for sym_raw, bars in bars_obj.items():
                sym = str(sym_raw).strip().upper()
                if not isinstance(bars, list):
                    continue
                out[sym] = []
                for rec in bars:
                    if isinstance(rec, dict):
                        row = dict(rec)
                        row.setdefault("symbol", sym)
                        out[sym].append(row)
            return out

        if isinstance(bars_obj, list):
            for rec in bars_obj:
                if not isinstance(rec, dict):
                    continue
                sym = str(rec.get("S") or rec.get("symbol") or "").strip().upper()
                if not sym:
                    continue
                out.setdefault(sym, []).append(dict(rec))
            return out

        raise AlpacaAPIError(f"Unexpected bars payload type: {type(bars_obj).__name__}")

    @staticmethod
    def _bar_sort_key(rec: Dict[str, Any]) -> Tuple[str, str]:
        sym = str(rec.get("symbol") or rec.get("S") or "")
        ts = str(rec.get("t") or rec.get("timestamp") or "")
        return (sym, ts)

    @staticmethod
    def _retry_backoff_seconds(attempt: int, base_sec: float, max_sec: float) -> float:
        k = max(0, int(attempt))
        base = max(0.0, float(base_sec))
        cap = max(0.0, float(max_sec))
        sleep = base * float(2**k)
        return float(min(sleep, cap))

    @staticmethod
    def _is_invalid_limit_error(status_code: int, text: str) -> bool:
        if int(status_code) not in (400, 422):
            return False
        t = str(text or "").lower()
        if "limit" not in t:
            return False
        return any(k in t for k in ("invalid", "max", "must be", "exceeds"))
