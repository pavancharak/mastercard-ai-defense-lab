"""Fixed, synthetic product catalog for the mock merchant API. Every value
here is invented for this demo -- no real merchant, product, or price.

One clearly-best-matching product per scripted purchase intent (see
scenarios.py) so the live OpenAI agent's tool-calling reliably converges
on the intended purchase without needing open-ended reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Product:
    product_id: str
    name: str
    category: str
    price: float
    merchant_id: str
    merchant_name: str
    # Merchant-side metadata Defend's classifier uses as real signal (a
    # merchant legitimately knows its own registration age / reputation;
    # this is not mandate-adjacent data).
    merchant_registration_age_days: int
    merchant_reputation_score: float
    recurring: bool  # True for subscription-style products
    currency: str = "USD"
    stock: int = 999


CATALOG: list[Product] = [
    Product(
        product_id="prod_netflix",
        name="Netflix Premium Subscription",
        category="subscription",
        price=15.99,
        merchant_id="mrc_streamco",
        merchant_name="StreamCo",
        merchant_registration_age_days=3650,
        merchant_reputation_score=0.96,
        recurring=True,
    ),
    Product(
        product_id="prod_spotify",
        name="Spotify Premium Subscription",
        category="subscription",
        price=11.99,
        merchant_id="mrc_streamco",
        merchant_name="StreamCo",
        merchant_registration_age_days=3650,
        merchant_reputation_score=0.96,
        recurring=True,
    ),
    Product(
        product_id="prod_headphones",
        name="Wireless Noise-Cancelling Headphones",
        category="electronics",
        price=165.00,
        merchant_id="mrc_techmart",
        merchant_name="TechMart",
        merchant_registration_age_days=2410,
        merchant_reputation_score=0.91,
        recurring=False,
    ),
    Product(
        product_id="prod_cloudsafe_annual",
        name="CloudSafe Storage Plan (Premium Annual Tier)",
        category="subscription",
        price=250.00,
        merchant_id="mrc_cloudsafe",
        merchant_name="CloudSafe",
        merchant_registration_age_days=1820,
        merchant_reputation_score=0.88,
        recurring=True,
    ),
    Product(
        product_id="prod_grocery_giftcard",
        name="$45 Grocery Gift Card",
        category="grocery",
        price=45.00,
        merchant_id="mrc_freshmart",
        merchant_name="FreshMart",
        merchant_registration_age_days=4100,
        merchant_reputation_score=0.94,
        recurring=False,
    ),
    Product(
        product_id="prod_pixelsuite",
        name="PixelSuite Premium Subscription (Monthly)",
        category="subscription",
        price=185.00,
        merchant_id="mrc_pixelsuite",
        merchant_name="PixelSuite",
        merchant_registration_age_days=980,
        merchant_reputation_score=0.83,
        recurring=True,
    ),
]

CATALOG_BY_ID: dict[str, Product] = {p.product_id: p for p in CATALOG}


def list_products(category: str | None = None) -> list[Product]:
    if category is None:
        return list(CATALOG)
    return [p for p in CATALOG if p.category == category]


def get_product(product_id: str) -> Product | None:
    return CATALOG_BY_ID.get(product_id)
