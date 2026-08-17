\# Attack Taxonomy: GenAI Powered Payment Fraud



Identify pillar, Mastercard Innovation Challenge, AI Defense Lab for Payment Security.



Fourteen attack vectors across five categories, each grounded in documented real world mechanism, current as of mid 2026. Each entry states the mechanism, the channel or rail it targets, the transaction level signal it should leave behind for a detection model to learn from, and a plausibility and severity rating. This taxonomy is the shared input for the Generate and Defend pillars: every vector listed here should have a corresponding simulated instance in Generate's dataset, and Defend's evaluation should report performance broken out by vector, not just in aggregate.



\## Category A: Identity and Onboarding Fraud



\### A1. Synthetic Identity Construction

\*\*Mechanism\*\*: An attacker combines a real, often stolen or purchased, identity fragment (a valid Social Security number or PAN, a real address history) with fabricated elements (a generated name, a GenAI produced face, a fabricated employment history) to build a composite identity that passes initial onboarding checks. Generative AI lowers the cost of the fabricated elements from requiring a skilled forger to requiring a prompt.

\*\*Channel\*\*: Account opening, card application, BNPL onboarding.

\*\*Transaction signal\*\*: No prior credit history despite claimed age or employment tenure; thin-file behavior with unusually mature spending patterns from the first transaction; identity attributes that individually pass verification but show low cross-source correlation.

\*\*Plausibility\*\*: Very high. Reported as the fastest growing onboarding fraud category through 2026, with tens of billions in annual losses attributed to this category in the US alone.

\*\*Severity\*\*: High. Synthetic identities are frequently used as durable infrastructure for later money movement, not just a single fraudulent account.



\### A2. Deepfake Document and Video KYC Bypass

\*\*Mechanism\*\*: GenAI generates a fabricated government ID (driver's license, passport) with realistic surface features such as holographic textures and correct shadowing, or a real time deepfake video substituted during a liveness check, defeating passive image quality based verification.

\*\*Channel\*\*: Digital account opening, high value transaction step up verification.

\*\*Transaction signal\*\*: Verification completed but with device or session characteristics inconsistent with a live capture (frame rate anomalies, injected video artifacts, mismatched metadata); onboarding velocity inconsistent with a genuine document retrieval and capture flow.

\*\*Plausibility\*\*: High, and growing quickly as deepfake generation quality improves faster than passive detection.

\*\*Severity\*\*: High. This is the entry point that enables several other categories below.



\## Category B: Social Engineering and Authorization Fraud



\### B1. Deepfake Voice Executive Impersonation (Business Email Compromise variant)

\*\*Mechanism\*\*: A cloned voice, built from a short public sample (an earnings call, a conference talk), is used in a live or voicemail call instructing a finance employee to execute an urgent wire transfer, often paired with a spoofed or lookalike email thread for corroboration.

\*\*Channel\*\*: Wire transfer, ACH, high value B2B payment.

\*\*Transaction signal\*\*: First-time or rare payee, urgency framing inconsistent with the payer's historical approval pattern, request originating or approved outside normal business hours or normal approval hierarchy.

\*\*Plausibility\*\*: High and rising; reported to have affected a meaningful share of large enterprises.

\*\*Severity\*\*: Very high per incident given typical wire transfer amounts and the low reversibility of the rail.



\### B2. Deepfake Video Authorized Push Payment Fraud

\*\*Mechanism\*\*: A real time or pre recorded deepfake video call, sometimes impersonating law enforcement, a bank fraud department, or a trusted contact, psychologically pressures a consumer into authorizing a payment or providing a one time passcode, exploiting the fact that the payment itself is genuinely authorized by the actual accountholder.

\*\*Channel\*\*: Real time payments, P2P transfer, instant bank transfer.

\*\*Transaction signal\*\*: This category is the hardest to catch on transaction data alone since authentication succeeds; signal must come from behavioral context: a first time large transfer to a new payee shortly after a support or verification style call, unusual transfer timing relative to account history.

\*\*Plausibility\*\*: High; documented at significant scale in some markets.

\*\*Severity\*\*: High, and specifically difficult for a purely transaction level classifier, worth naming explicitly in the Defend pillar's known limitations rather than claiming full coverage.



\### B3. Multivector AI Personalized Phishing and Vishing

\*\*Mechanism\*\*: An LLM personalizes phishing content (email, SMS, voice script) using scraped professional or social data, then adapts the script in real time to a victim's responses across multiple channels in a single coordinated campaign, harvesting credentials or card data.

\*\*Channel\*\*: Any, feeds into card not present fraud or account takeover downstream.

\*\*Transaction signal\*\*: Not directly visible in a single transaction; the downstream signal is credential use with a session or device profile inconsistent with the account's history immediately after a plausible phishing exposure window.

\*\*Plausibility\*\*: Very high; documented as one of the largest and fastest growing GenAI enabled categories, with a large majority of phishing content now containing GenAI generated material.

\*\*Severity\*\*: Moderate to high per incident, very high in aggregate given volume.



\## Category C: Agentic Commerce Fraud



This category is newly emerging in 2026 and directly relevant to Mastercard specifically, given the company's own public work rebuilding fraud logic to accommodate autonomous purchasing agents. Judges from this specific sponsor are likely to weight this category's inclusion favorably.



\### C1. Agent Impersonation

\*\*Mechanism\*\*: A malicious actor spoofs the identity or credentials of a legitimate AI shopping or payment agent to gain a merchant's or network's trust, then executes fraudulent purchases using the trust extended to genuine agentic traffic.

\*\*Channel\*\*: Agentic checkout, delegated payment mandate flows.

\*\*Transaction signal\*\*: Agent-attributed transaction with a credential or attestation that is present but does not independently verify (this maps directly onto the kind of check Mastercard's own Verifiable Intent work and Visa's Trusted Agent Protocol are designed to strengthen); mismatch between claimed agent identity and observed transaction behavior pattern for that agent.

\*\*Plausibility\*\*: Emerging, expected to grow quickly as agentic commerce adoption scales; a majority of merchants report actively exploring agentic payment acceptance.

\*\*Severity\*\*: High and structurally novel, since it targets the trust layer being built right now rather than an established, already defended surface.



\### C2. Malicious Storefront Targeting Legitimate Agents

\*\*Mechanism\*\*: An attacker sets up a fraudulent storefront or injects content designed to deceive a legitimate consumer facing shopping agent into completing a purchase against a fraudulent or non delivering merchant, exploiting the agent's reduced human skepticism compared to a person.

\*\*Channel\*\*: Agentic checkout, card not present.

\*\*Transaction signal\*\*: Agent initiated transaction to a newly registered or low reputation merchant, transaction characteristics inconsistent with the consumer's own historical merchant preferences.

\*\*Plausibility\*\*: Emerging, directly named as a documented pattern in network level analysis of the agentic commerce threat landscape.

\*\*Severity\*\*: Moderate per incident, high as a category given how fast agentic shopping traffic is growing.



\### C3. Delegated Mandate Scope Abuse

\*\*Mechanism\*\*: A legitimate agent, or an attacker who has compromised one, executes a transaction that technically falls within a broad delegated mandate from the consumer but violates the actual intent (wrong amount, wrong merchant category, unauthorized recurring charge), exploiting ambiguity between what was authorized and what was executed. This is structurally the same class of gap as an authorization boundary that does not bind the specific execution content to what was actually approved, applied to consumer facing agentic commerce rather than enterprise execution infrastructure.

\*\*Channel\*\*: Agentic checkout, subscription and recurring agent driven payments.

\*\*Transaction signal\*\*: Transaction content diverging from the pattern of prior transactions executed under the same mandate; amount or merchant category outside the historical envelope for that agent-consumer pairing.

\*\*Plausibility\*\*: Emerging; a direct consequence of agentic payment infrastructure still being standardized in 2026.

\*\*Severity\*\*: Moderate to high, and one of the more defensible novel contributions this submission can make given it connects directly to authorization-boundary thinking rather than only classification.



\## Category D: Automated, Machine Speed Attacks



\### D1. AI Accelerated Card Testing

\*\*Mechanism\*\*: Stolen or generated card numbers are validated against live payment endpoints at high speed and low cost using AI assisted automation, replacing what previously required manual or scripted trial and error, to determine which stolen credentials are usable before deploying them for larger fraud.

\*\*Channel\*\*: Card not present, low value test transactions across many merchants.

\*\*Transaction signal\*\*: Very low value, high velocity transaction attempts across a cluster of merchants or a single merchant in a short window, often with low authorization success rate followed by a spike in usage of successfully validated cards elsewhere.

\*\*Plausibility\*\*: Very high; documented as rising sharply year over year through 2026.

\*\*Severity\*\*: Moderate per transaction, high as an early warning signal, since detecting this category early can prevent the downstream fraud it enables.



\### D2. AI Scaled Credential Stuffing and Account Takeover

\*\*Mechanism\*\*: Previously breached credentials are tested against payment and banking accounts at scale using AI assisted automation that adapts to rate limiting and CAPTCHA style defenses more effectively than earlier scripted bots.

\*\*Channel\*\*: Account login preceding a payment or transfer action.

\*\*Transaction signal\*\*: Login from a new device or geography immediately followed by a payment or transfer, especially one that changes account contact details or payout destinations first.

\*\*Plausibility\*\*: Very high; documented as surging sharply through 2026.

\*\*Severity\*\*: High, since successful account takeover typically enables several of the categories above in sequence.



\## Category E: Post Transaction and Dispute Fraud



\### E1. GenAI Fabricated Refund and Chargeback Evidence

\*\*Mechanism\*\*: An attacker generates fabricated but visually convincing evidence, a fake damaged product photo, a fabricated receipt with correct merchant branding, to support a false refund or chargeback claim, at a volume that would not have been economical to produce manually.

\*\*Channel\*\*: Post purchase dispute and refund flow.

\*\*Transaction signal\*\*: Refund or dispute filed with supporting media, correlated across multiple claims with subtle generative artifacts, unusually high dispute rate from an account with an otherwise clean transaction history.

\*\*Plausibility\*\*: High, and specifically noted as having overtaken payment fraud as the top ranked merchant concern in some 2026 industry reporting.

\*\*Severity\*\*: Moderate per incident, high in aggregate; this category is a strong candidate for the Defend pillar to cover if time allows a second model or ruleset, since it is a distinct data shape from point of sale fraud (media plus dispute metadata rather than pure transaction fields).



\### E2. Promotion and Coupon Abuse at GenAI Scale

\*\*Mechanism\*\*: Automated generation of plausible but distinct synthetic identities or accounts to repeatedly claim promotional discounts, referral bonuses, or new customer offers beyond what any single legitimate customer could claim.

\*\*Channel\*\*: Promotional and loyalty program redemption tied to a payment.

\*\*Transaction signal\*\*: Cluster of accounts with distinct identities but correlated device, network, or behavioral fingerprints, each making a single low value promotional transaction shortly after account creation.

\*\*Plausibility\*\*: High, documented as a persistent and growing abuse category.

\*\*Severity\*\*: Low to moderate per incident, but high aggregate cost to merchants running promotional programs at scale.



\## Notes for the Generate and Defend pillars



\- Categories C (Agentic Commerce Fraud) and D (Automated, Machine Speed Attacks) are the strongest candidates for the primary detection model, since they produce the clearest transaction level signal and are the most current and Mastercard relevant. Recommend Defend prioritize these first if time forces a choice.

\- Category B (Social Engineering and Authorization Fraud) is explicitly hard to detect from transaction data alone, since the payment itself is genuinely authorized by the real accountholder. Recommend the solution walkthrough name this limitation honestly rather than implying full coverage, and, if time allows, propose behavioral or contextual features (payee novelty, timing relative to a support interaction) as a stated direction rather than a fully built capability.

\- Category A (Identity and Onboarding Fraud) sits mostly upstream of the transaction level, at account opening rather than at payment time. Generate should still simulate its downstream transaction signature (thin file with mature spending pattern) since that is what a payment level detector could realistically observe.

\- Category E (Post Transaction and Dispute Fraud) requires a different data shape than the other four categories, transaction plus dispute metadata plus, ideally, some representation of submitted evidence, rather than pure point of sale fields. Worth a deliberate decision on whether to include it in the primary model or flag it as a documented extension given time constraints.

