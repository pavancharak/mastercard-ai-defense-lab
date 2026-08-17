"""A minimal, entirely local, synthetic merchant API.

This is the ONLY system the agent is permitted to call. It knows nothing
about any consumer's mandate -- it just fulfills orders the way a real
merchant would (does the product exist, is it in stock), which is
exactly why "the purchase succeeded at the merchant level" is a
necessary-but-not-sufficient signal and needs the two other, independent
checks (classifier.py, mandate.py) alongside it.

Runs on 127.0.0.1 only (see server.py) -- never bound to any interface
reachable off this machine, and never itself makes an outbound call to
anything.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .catalog import get_product, list_products

app = FastAPI(title="Synthetic Mock Merchant API (local demo only)")


class ProductOut(BaseModel):
    product_id: str
    name: str
    category: str
    price: float
    currency: str
    merchant_id: str
    merchant_name: str
    recurring: bool
    in_stock: bool


class PurchaseRequest(BaseModel):
    product_id: str
    quantity: int = 1
    recurring: bool | None = None  # None = use the product's own default


class PurchaseResult(BaseModel):
    success: bool
    transaction_id: str | None = None
    decline_reason: str | None = None
    product_id: str
    product_name: str | None = None
    category: str | None = None
    amount: float | None = None
    currency: str | None = None
    merchant_id: str | None = None
    merchant_name: str | None = None
    recurring: bool | None = None
    merchant_registration_age_days: int | None = None
    merchant_reputation_score: float | None = None
    timestamp: str


@app.get("/health")
def health() -> dict:
    return {"status": "up", "note": "synthetic local mock merchant, no real payment rails"}


@app.get("/products", response_model=list[ProductOut])
def get_products(category: str | None = None) -> list[ProductOut]:
    products = list_products(category)
    return [
        ProductOut(
            product_id=p.product_id,
            name=p.name,
            category=p.category,
            price=p.price,
            currency=p.currency,
            merchant_id=p.merchant_id,
            merchant_name=p.merchant_name,
            recurring=p.recurring,
            in_stock=p.stock > 0,
        )
        for p in products
    ]


@app.get("/products/{product_id}", response_model=ProductOut)
def get_single_product(product_id: str) -> ProductOut:
    """Idempotent, read-only re-verification that a given product still
    exists and is in stock -- used as the independent "does this still
    succeed at the merchant level" check run alongside (not instead of)
    the classifier and mandate-envelope checks. Never mutates state."""

    product = get_product(product_id)
    if product is None:
        raise HTTPException(status_code=404, detail=f"unknown product_id '{product_id}'")
    return ProductOut(
        product_id=product.product_id,
        name=product.name,
        category=product.category,
        price=product.price,
        currency=product.currency,
        merchant_id=product.merchant_id,
        merchant_name=product.merchant_name,
        recurring=product.recurring,
        in_stock=product.stock > 0,
    )


@app.post("/purchase", response_model=PurchaseResult)
def submit_purchase(req: PurchaseRequest) -> PurchaseResult:
    """Merchant-level check only: does this product exist and is it in
    stock. Deliberately has no concept of "is this within the buyer's
    delegated mandate" -- that's not a merchant's job, it's the mandate
    envelope check's."""

    timestamp = datetime.now(UTC).isoformat()
    product = get_product(req.product_id)

    if product is None:
        return PurchaseResult(
            success=False,
            decline_reason=f"unknown product_id '{req.product_id}'",
            product_id=req.product_id,
            timestamp=timestamp,
        )

    if product.stock <= 0:
        return PurchaseResult(
            success=False,
            decline_reason="out of stock",
            product_id=req.product_id,
            product_name=product.name,
            timestamp=timestamp,
        )

    recurring = product.recurring if req.recurring is None else req.recurring

    return PurchaseResult(
        success=True,
        transaction_id=f"txn_{uuid.uuid4().hex[:12]}",
        product_id=product.product_id,
        product_name=product.name,
        category=product.category,
        amount=round(product.price * max(req.quantity, 1), 2),
        currency=product.currency,
        merchant_id=product.merchant_id,
        merchant_name=product.merchant_name,
        recurring=recurring,
        merchant_registration_age_days=product.merchant_registration_age_days,
        merchant_reputation_score=product.merchant_reputation_score,
        timestamp=timestamp,
    )
