"""Shared synthetic population that vector generators draw from.

Keeps IDs, merchant categories, and account histories coherent across the
dataset (e.g. an account takeover in D2 reuses a real established account
from this pool rather than inventing an unrelated one), without building a
full relational transaction graph, which would be overkill for a labeled
training set where the signal that matters is per-row feature distributions.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import numpy as np
from faker import Faker

from .config import GEOS, MERCHANT_CATEGORIES


def short_id(prefix: str, rng: np.random.Generator) -> str:
    # rng.integers(0, 2**63) only ever fills the low 63 bits of the 128-bit
    # UUID integer; .hex is big-endian, so the *leading* hex digits are
    # always zero and the randomness lives in the *trailing* ones. Slicing
    # from the end (not the start) is what actually gives 10 hex digits of
    # real entropy (40 bits) per ID instead of a constant string.
    return f"{prefix}_{uuid.UUID(int=int(rng.integers(0, 2**63))).hex[-10:]}"


@dataclass
class Account:
    account_id: str
    customer_id: str
    customer_name: str
    account_age_days: int
    prior_transaction_count: int
    historical_avg_amount: float
    geo_country: str
    device_id: str
    credit_history_months: int


@dataclass
class Merchant:
    merchant_id: str
    name: str
    category: str
    registration_age_days: int
    reputation_score: float
    is_malicious: bool = False


@dataclass
class Agent:
    agent_id: str
    linked_account_id: str
    linked_account: Account
    mandate_id: str
    mandate_amount_cap: float
    mandate_category_scope: str
    mandate_recurring_flag: bool


@dataclass
class EntityPool:
    accounts: list[Account] = field(default_factory=list)
    merchants: list[Merchant] = field(default_factory=list)
    malicious_merchants: list[Merchant] = field(default_factory=list)
    agents: list[Agent] = field(default_factory=list)

    def sample_account(self, rng: np.random.Generator) -> Account:
        return self.accounts[rng.integers(0, len(self.accounts))]

    def sample_merchant(self, rng: np.random.Generator) -> Merchant:
        return self.merchants[rng.integers(0, len(self.merchants))]

    def sample_malicious_merchant(self, rng: np.random.Generator) -> Merchant:
        return self.malicious_merchants[rng.integers(0, len(self.malicious_merchants))]

    def sample_agent(self, rng: np.random.Generator) -> Agent:
        return self.agents[rng.integers(0, len(self.agents))]


def build_entity_pool(
    rng: np.random.Generator,
    n_accounts: int = 4_000,
    n_merchants: int = 500,
    n_malicious_merchants: int = 40,
    n_agents: int = 150,
) -> EntityPool:
    faker = Faker()
    Faker.seed(int(rng.integers(0, 2**32 - 1)))

    def _make_account() -> Account:
        # No floor: real accounts get used within days of opening constantly,
        # so a legit population that's never younger than 30 days (the old
        # bound) is itself unrealistic - the same class of issue as the
        # pre-fix wire-amount ceiling. A brand-new account also genuinely
        # has no transaction history yet, so historical_avg_amount/
        # prior_transaction_count are forced to 0 below that same young-age
        # threshold rather than independently drawn - an account 2 days old
        # showing a $340 "historical average" would itself be a tell.
        age = int(rng.integers(0, 4000))
        is_brand_new = age < 3
        return Account(
            account_id=short_id("acct", rng),
            customer_id=short_id("cust", rng),
            customer_name=faker.name(),
            account_age_days=age,
            prior_transaction_count=0 if is_brand_new else int(rng.integers(0, 800)),
            historical_avg_amount=0.0 if is_brand_new else float(np.round(rng.lognormal(mean=4.0, sigma=0.8), 2)),
            geo_country=rng.choice(GEOS),
            device_id=short_id("dev", rng),
            credit_history_months=int(rng.integers(0, 240)),
        )

    accounts = [_make_account() for _ in range(n_accounts)]

    merchants = [
        Merchant(
            merchant_id=short_id("mrc", rng),
            name=faker.company(),
            category=rng.choice(MERCHANT_CATEGORIES),
            registration_age_days=int(rng.integers(180, 5000)),
            reputation_score=float(np.round(rng.beta(8, 2), 3)),
            is_malicious=False,
        )
        for _ in range(n_merchants)
    ]

    malicious_merchants = [
        Merchant(
            merchant_id=short_id("mrc", rng),
            name=faker.company() + " " + faker.bs(),
            category=rng.choice(MERCHANT_CATEGORIES),
            registration_age_days=int(rng.integers(0, 30)),
            reputation_score=float(np.round(rng.beta(1.2, 6), 3)),
            is_malicious=True,
        )
        for _ in range(n_malicious_merchants)
    ]

    agents = []
    for _ in range(n_agents):
        linked = accounts[rng.integers(0, len(accounts))]
        agents.append(
            Agent(
                agent_id=short_id("agt", rng),
                linked_account_id=linked.account_id,
                linked_account=linked,
                mandate_id=short_id("mnd", rng),
                mandate_amount_cap=float(np.round(rng.lognormal(mean=4.5, sigma=0.6), 2)),
                mandate_category_scope=rng.choice(MERCHANT_CATEGORIES),
                mandate_recurring_flag=bool(rng.random() < 0.3),
            )
        )

    return EntityPool(
        accounts=accounts,
        merchants=merchants,
        malicious_merchants=malicious_merchants,
        agents=agents,
    )
