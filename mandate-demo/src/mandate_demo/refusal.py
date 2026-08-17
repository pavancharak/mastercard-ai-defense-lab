"""Structured refusal records, shaped to mirror Parmana's real
RefusalRecord (python/parmana/models/refusal_record.py in parmana-exp,
generated from packages/shared/src/domain/refusal-record.ts) -- read for
reference, not imported: this module contains no Parmana code and no
call to Parmana infrastructure.

Field-for-field mapping to the real shape:
  refusal_record_id        <-> refusal_record_id
  purchase_attempt_id       <-> business_transaction_id
  decision                  <-> decision
  evaluated_intent          <-> evaluated_intent (target/parameters)
  mandate_binding_violations <-> binding_violations (signal_key/intent_path/
                                 signal_value/intent_value renamed to
                                 field/mandate_authorized/actually_proposed
                                 for this demo's own vocabulary)
  record_hash               <-> refusal_record_hash
  created_at                <-> created_at
  submitted_by              <-> submitted_by

One deliberate, honestly-labeled omission: the real RefusalRecord also
carries a cryptographic `signature` over the record, produced by
Parmana's real signing stack. This demo has no key-management or signing
infrastructure, so `record_hash` here is a plain SHA-256 over the
record's canonical JSON for tamper-evidence display only -- it is
explicitly NOT a signature and must never be presented as one.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

from .mandate import BindingViolation


@dataclass(frozen=True)
class EvaluatedIntent:
    target: str  # merchant/product being purchased
    parameters: dict[str, object]


@dataclass(frozen=True)
class RefusalRecord:
    refusal_record_id: str
    purchase_attempt_id: str
    decision: str  # "REFUSED"
    evaluated_intent: EvaluatedIntent
    mandate_binding_violations: list[BindingViolation]
    submitted_by: str
    created_at: str
    record_hash: str
    signature: str = "UNSIGNED -- local demo, no signing key material"


def _canonical_json(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))


def build_refusal_record(
    *,
    refusal_record_id: str,
    purchase_attempt_id: str,
    target: str,
    parameters: dict[str, object],
    violations: list[BindingViolation],
    submitted_by: str,
) -> RefusalRecord:
    evaluated_intent = EvaluatedIntent(target=target, parameters=parameters)
    created_at = datetime.now(UTC).isoformat()

    unhashed = {
        "refusal_record_id": refusal_record_id,
        "purchase_attempt_id": purchase_attempt_id,
        "decision": "REFUSED",
        "evaluated_intent": asdict(evaluated_intent),
        "mandate_binding_violations": [asdict(v) for v in violations],
        "submitted_by": submitted_by,
        "created_at": created_at,
    }
    record_hash = hashlib.sha256(_canonical_json(unhashed).encode("utf-8")).hexdigest()

    return RefusalRecord(
        refusal_record_id=refusal_record_id,
        purchase_attempt_id=purchase_attempt_id,
        decision="REFUSED",
        evaluated_intent=evaluated_intent,
        mandate_binding_violations=violations,
        submitted_by=submitted_by,
        created_at=created_at,
        record_hash=record_hash,
    )


def refusal_record_to_dict(record: RefusalRecord) -> dict:
    return asdict(record)
