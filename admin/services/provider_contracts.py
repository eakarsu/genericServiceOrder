from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from typing import Protocol

import requests


def payload_digest(raw_body: bytes) -> str:
    return hashlib.sha256(raw_body).hexdigest()


def verify_webhook_signature(raw_body: bytes, signature: str | None, secret: str) -> bool:
    if not signature or not secret:
        return False
    supplied = signature.removeprefix("sha256=")
    expected = hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(supplied, expected)


@dataclass(frozen=True)
class TaxQuote:
    subtotal_cents: int
    tax_cents: int
    jurisdiction: str


@dataclass(frozen=True)
class CalendarReservation:
    external_id: str
    status: str


@dataclass(frozen=True)
class DeliveryReceipt:
    external_id: str
    status: str


class ProviderContracts(Protocol):
    """Typed seam for maps, calendar, messaging, tax, payment, and accounting adapters."""

    def travel_minutes(self, origin: str, destination: str) -> int: ...
    def calculate_tax(self, subtotal_cents: int, service_area_code: str) -> TaxQuote: ...
    def reserve_calendar(self, idempotency_key: str, resource_id: str, start: str, end: str) -> CalendarReservation: ...
    def send_message(self, idempotency_key: str, channel: str, payload: dict) -> DeliveryReceipt: ...
    def create_payment(self, idempotency_key: str, invoice_id: str, amount_cents: int) -> str: ...
    def post_invoice(self, idempotency_key: str, invoice: dict) -> str: ...


class HttpProviderContracts:
    """HTTP adapter for a provider gateway with bounded calls and idempotency keys."""

    def __init__(self, urls: dict[str, str], api_token: str, timeout_seconds: float = 5.0):
        missing = [name for name in ("maps", "calendar", "messaging", "payment", "tax", "accounting") if not urls.get(name)]
        if missing or not api_token:
            raise ValueError("Provider URLs and API token are required: " + ", ".join(missing))
        self.urls = urls
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_token}", "Accept": "application/json"})

    def _post(self, provider: str, payload: dict, idempotency_key: str | None = None) -> dict:
        headers = {"Idempotency-Key": idempotency_key} if idempotency_key else None
        response = self.session.post(
            self.urls[provider], json=payload, headers=headers, timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        value = response.json()
        if not isinstance(value, dict):
            raise ValueError(f"{provider} provider returned a non-object response")
        return value

    def travel_minutes(self, origin: str, destination: str) -> int:
        value = self._post("maps", {"origin": origin, "destination": destination})
        minutes = int(value["travel_minutes"])
        if minutes < 0 or minutes > 24 * 60:
            raise ValueError("Maps provider returned an invalid travel duration")
        return minutes

    def calculate_tax(self, subtotal_cents: int, service_area_code: str) -> TaxQuote:
        value = self._post("tax", {"subtotal_cents": subtotal_cents, "service_area_code": service_area_code})
        tax = int(value["tax_cents"])
        if tax < 0:
            raise ValueError("Tax provider returned a negative amount")
        return TaxQuote(subtotal_cents, tax, str(value["jurisdiction"]))

    def reserve_calendar(self, idempotency_key: str, resource_id: str, start: str, end: str) -> CalendarReservation:
        value = self._post("calendar", {"resource_id": resource_id, "start": start, "end": end}, idempotency_key)
        return CalendarReservation(str(value["external_id"]), str(value["status"]))

    def send_message(self, idempotency_key: str, channel: str, payload: dict) -> DeliveryReceipt:
        value = self._post("messaging", {"channel": channel, "payload": payload}, idempotency_key)
        return DeliveryReceipt(str(value["external_id"]), str(value["status"]))

    def create_payment(self, idempotency_key: str, invoice_id: str, amount_cents: int) -> str:
        value = self._post("payment", {"invoice_id": invoice_id, "amount_cents": amount_cents}, idempotency_key)
        return str(value["external_id"])

    def post_invoice(self, idempotency_key: str, invoice: dict) -> str:
        value = self._post("accounting", invoice, idempotency_key)
        return str(value["external_id"])


def decode_payment_event(raw_body: bytes) -> dict:
    try:
        value = json.loads(raw_body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Webhook body must be valid JSON") from exc
    required = {"event_id", "event_type", "order_id", "amount_cents"}
    if not isinstance(value, dict) or not required.issubset(value):
        raise ValueError("Webhook event is missing required fields")
    return value
