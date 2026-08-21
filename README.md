
````markdown
# Mastercard AI Defense Lab

## AI-era payment fraud: find it, create it, detect it, and stop it

A working system for testing payment fraud defenses against new AI-enabled attacks.

The system follows a simple loop:

**Identify → Generate → Detect → Enforce**

It first identifies new fraud patterns, creates realistic test transactions from those patterns, tests a fraud detector against them, and then applies an independent authorization check where detecting fraud from transaction patterns is not enough.

The complete system is available as a live web prototype.

## Live prototype

**https://mastercard-ai-defense-lab.fly.dev/**

Open the prototype and start with **Mandate Demo**.

The Mandate Demo shows an important limitation of fraud detection:

> A transaction can look normal and still be something the customer never authorized.

In the live demonstration, four transactions are refused by the mandate check even though the trained fraud model considers them low risk.

The prototype also includes:

- **Taxonomy** — the fraud attacks identified by the system
- **Dashboard** — detection results
- **Case Browser** — score transactions using the trained model
- **Mandate Demo** — show where fraud detection alone is not enough

---

# The problem

AI is making payment fraud faster and easier to automate.

An attacker can now create convincing identities, messages, transaction patterns and automated behaviour at scale.

Traditional fraud detection mainly asks:

> "Does this transaction look like fraud?"

That is useful, but it does not answer another important question:

> "Is this transaction actually allowed?"

Our system addresses both questions.

### 1. Does the transaction look fraudulent?

The **Defend** layer uses a trained fraud detector.

### 2. Is the action actually allowed?

The **Mandate** layer checks the proposed action against the actual authorization given to the customer or account.

This second check does not depend on a machine-learning score.

---

# How the system works

```text
             IDENTIFY
                 │
                 ▼
        Fraud attack patterns
                 │
                 ▼
             GENERATE
                 │
                 ▼
        Realistic test data
                 │
                 ▼
              DEFEND
                 │
          ┌──────┴──────┐
          ▼             ▼
       Detected       Missed
          │             │
          │             ▼
          │          MANDATE
          │             │
          │             ▼
          │       Is this action
          │        actually allowed?
          │             │
          └─────────────┘
````

The important point is that the system does not rely on one model for everything.

A statistical fraud detector looks for patterns.

An authorization check verifies what was actually allowed.

---

# 1. Identify

The `identify/` directory contains the fraud attack taxonomy.

It currently defines **12 attack vectors across five groups**:

* Identity and onboarding fraud
* Social engineering and authorization fraud
* Agentic commerce fraud
* Automated machine-speed attacks
* Post-transaction and loyalty fraud

Each attack describes:

* how the attack works
* where it happens
* what evidence it should leave behind
* how realistic and serious it is

The taxonomy is the specification used by the rest of the system.

File:

```text
identify/attack-taxonomy.md
```

The web prototype also provides a browsable version.

---

# 2. Generate

The `generate/` directory turns the attack descriptions into test data.

For each attack, the system creates synthetic transactions that resemble realistic payment activity.

It also creates legitimate transactions so that fraud and normal activity can be compared using the same structure.

Current dataset:

* **21,960 total records**
* **700 fraud records**
* 11 transaction-based attack vectors used by the current detector
* A separate dispute record structure for E1

The generation process has been tested for data leakage.

For example, an earlier version allowed the model to identify the generator rather than the actual fraud behaviour. Those issues were found and fixed, with tests added to prevent them from returning.

---

# 3. Defend

The `defend/` directory trains the fraud detector using the generated data.

The model is an XGBoost classifier.

The important part is not simply the overall score.

Results are also reported separately for each attack group and each attack vector.

## Test results

The held-out test set contains:

* **5,490 transactions**
* **175 fraud transactions**

Results:

| Metric    | Result |
| --------- | -----: |
| Precision |  0.944 |
| Recall    |  0.966 |
| F1        |  0.955 |
| ROC AUC   |  0.999 |
| PR AUC    |  0.985 |

The detector performs strongly overall, but the system does not hide where it is weaker.

## Recall by attack

| Attack | Recall |
| ------ | -----: |
| A1     |  1.000 |
| A2     |  1.000 |
| B1     |  0.867 |
| B2     |  0.800 |
| B3     |  0.960 |
| C1     |  1.000 |
| C2     |  1.000 |
| C3     |  1.000 |
| D1     |  1.000 |
| D2     |  1.000 |
| E2     |  1.000 |

The weaker results in the B group are important.

Some attacks are difficult to identify from transaction behaviour alone because the transaction can be genuinely authorized by the account holder.

That leads to the fourth part of the system.

---

# 4. Mandate: checking what was actually allowed

The `mandate-demo/` directory demonstrates a different type of protection.

Instead of asking:

> "Does this transaction look fraudulent?"

it asks:

> "Does this transaction follow the customer's actual authorization?"

The demonstration uses a synthetic purchasing agent and synthetic merchants.

The agent operates under a fixed mandate such as:

```text
Up to $200 per month on subscription services.
```

Each proposed purchase is checked three ways:

1. Can the merchant accept it?
2. What does the fraud detector say?
3. Does the transaction follow the actual mandate?

The third check is deterministic.

It does not use a fraud probability.

It does not need training data.

It does not need a threshold.

It simply checks the proposed action against the authorization record.

---

# Why this matters

Consider a $165 purchase.

The amount is within the customer's $200 limit.

The account looks established.

The transaction does not look unusual.

The fraud detector therefore gives it a very low fraud probability.

But if the purchase is outside the merchant category allowed by the customer's mandate, it must still be refused.

The fraud model can miss this because the transaction looks statistically normal.

The mandate check does not have to guess.

It checks the rule directly.

---

# Live Mandate Demo

The demonstration contains six purchase scenarios.

Results:

* **2 correctly allowed**
* **4 correctly refused**
* All four refused transactions were independently scored as low risk by the fraud classifier

One scenario exceeds the customer's monthly cumulative limit.

That case cannot be detected from the current transaction model because the model does not contain the customer's running spend under the mandate.

This is an important finding rather than something hidden from the evaluation.

It shows where a fraud model reaches the limit of what transaction patterns can tell us.

---

# The closed loop

The system is designed to improve the defense by testing it against attacks it has not seen before.

```text
New attack
    ↓
Identify the behaviour
    ↓
Generate realistic examples
    ↓
Test the detector
    ↓
Find what it misses
    ↓
Add another layer of protection
    ↓
Test again
```

The goal is therefore not simply to build a model with a high score.

The goal is to discover where the defense fails.

---

# Web prototype

The `web-prototype/` directory provides the complete demonstration.

It connects:

* the attack taxonomy
* the generated dataset
* the trained fraud model
* the mandate checks

The prototype provides five main views:

1. Overview
2. Attack Taxonomy
3. Detection Dashboard
4. Case Browser
5. Mandate Demo

The Case Browser can score transactions from the generated dataset or accept transaction information manually.

The Dashboard displays the actual model results rather than manually entered demonstration numbers.

---

# Repository structure

```text
mastercard-ai-defense-lab/
│
├── identify/
│   └── attack-taxonomy.md
│
├── generate/
│   ├── data/
│   ├── src/
│   └── tests/
│
├── defend/
│   ├── results/
│   ├── src/
│   └── tests/
│
├── mandate-demo/
│   ├── output/
│   ├── src/
│   └── tests/
│
├── web-prototype/
│   ├── src/
│   └── tests/
│
├── Dockerfile
├── fly.toml
├── .env.example
└── README.md
```

---

# Run locally

Python 3.11 or newer is required.

## Generate the dataset

```bash
cd generate

python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"

.venv/Scripts/python -m pytest -q
.venv/Scripts/python -m generate.cli --format csv
```

## Train the detector

```bash
cd ../defend

python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"

.venv/Scripts/python -m pytest -q
.venv/Scripts/python -m defend.train
```

## Run the Mandate Demo

Set:

```text
OPENAI_API_KEY
```

Then:

```bash
cd ../mandate-demo

python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"

.venv/Scripts/python -m pytest -q
.venv/Scripts/python -m mandate_demo.runner
```

## Run the web prototype

```bash
cd ../web-prototype

python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"
.venv/Scripts/pip install -e ../mandate-demo

.venv/Scripts/python -m pytest -q

.venv/Scripts/python -m uvicorn web_prototype.app:app \
  --app-dir src \
  --host 127.0.0.1 \
  --port 8420
```

Open:

```text
http://127.0.0.1:8420
```

Or use the live deployment:

**[https://mastercard-ai-defense-lab.fly.dev/](https://mastercard-ai-defense-lab.fly.dev/)**

---

# What is demonstrated

This repository contains a working implementation of all three challenge pillars:

### Identify

A documented set of emerging payment fraud attacks.

### Generate

A synthetic dataset built from those attacks.

### Defend

A trained fraud detector evaluated on held-out data.

### Enforce

An independent authorization check for cases where statistical fraud detection is not enough.

### Demonstrate

A live web application connecting the complete system.

---

# What is intentionally not claimed

This is a research and demonstration system using synthetic payment data.

It does not connect to real card networks, banks, merchants or customer accounts.

The merchants, products, accounts and mandates used by the Mandate Demo are synthetic.

The system demonstrates the architecture and the control approach; it is not presented as a production payment network integration.

---

# Known limitations

We are deliberately documenting the current limitations.

### E1 is not included in the current model evaluation

The fabricated-dispute attack is present in the taxonomy and dataset structure but is excluded from the current detector evaluation so the reported results remain comparable to the earlier evaluation run.

### Two dataset issues remain

The generator still has:

* an account-age issue affecting four attack vectors
* two attack vectors where some fields do not yet have a matching legitimate counterpart

These are documented rather than hidden.

### The web form exposes only selected model fields

The free-form Case Browser exposes 19 of the model's 38 features.

Blank fields remain missing rather than being filled with invented values.

### No production account system

The prototype has no authentication, database or customer persistence.

It is designed as a judge-facing demonstration.

---

# Security and API keys

Never commit an API key.

For local development:

```text
.env
```

is used and excluded from the Docker build.

For the deployed application, the API key is stored using the deployment platform's secret management.

---

# The key idea

Fraud detection and authorization solve different problems.

A fraud model asks:

> **"Does this look suspicious?"**

Authorization asks:

> **"Is this actually allowed?"**

A strong payment defense needs both.

**Detect what looks wrong.
Verify what is allowed.
Stop what is not authorized.**


