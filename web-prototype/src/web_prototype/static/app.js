// Vanilla JS only. Every network call here goes to this same server's own
// /api/* endpoints -- never to OpenAI or any other external host. The
// server-side code (live_agent.py) is what actually calls OpenAI, using an
// API key read from the server process's own environment; that key is never
// sent to, or present in, any file this script loads or any response this
// script reads.

function formatPercent(x) {
  return (x * 100).toFixed(2) + "%";
}

function riskLabelHtml(riskLabel, fraudProbability) {
  const cls = riskLabel === "LOW_RISK" ? "risk-low" : "risk-high";
  return `<span class="${cls}">${riskLabel}</span> (fraud_probability = ${fraudProbability.toFixed(4)})`;
}

function renderScoreResult(container, data, extra) {
  container.classList.remove("hidden");
  let html = `<div class="result-headline">${riskLabelHtml(data.risk_label, data.fraud_probability)}</div>`;
  html += `<p class="muted">decision_threshold = ${data.decision_threshold}</p>`;
  if (data.ground_truth) {
    const gt = data.ground_truth;
    html += `<p><strong>Labeled ground truth (from Generate's dataset):</strong> `
      + `is_fraud=${gt.is_fraud}, fraud_category=${gt.fraud_category || "--"}, `
      + `fraud_vector=${gt.fraud_vector || "--"}</p>`;
    if (gt.narrative_tag) {
      html += `<p class="muted">${gt.narrative_tag}</p>`;
    }
  }
  if (extra) html += extra;
  html += `<details><summary class="muted">Full feature vector sent to the classifier</summary>`
    + `<pre style="white-space:pre-wrap;font-size:0.8rem;">${JSON.stringify(data.features_sent, null, 2)}</pre></details>`;
  container.innerHTML = html;
}

document.addEventListener("DOMContentLoaded", () => {
  // --- Case browser: dataset scoring ---
  const datasetResult = document.getElementById("dataset-result");

  async function scoreDataset(transactionId) {
    if (!datasetResult) return;
    datasetResult.classList.remove("hidden");
    datasetResult.innerHTML = `<p class="muted">Scoring ${transactionId}...</p>`;
    const form = new FormData();
    form.append("transaction_id", transactionId);
    const resp = await fetch("/api/score/dataset", { method: "POST", body: form });
    const data = await resp.json();
    if (!resp.ok) {
      datasetResult.innerHTML = `<p class="error">${data.error || "scoring failed"}</p>`;
      return;
    }
    renderScoreResult(datasetResult, data);
  }

  document.querySelectorAll(".score-dataset-btn").forEach((btn) => {
    btn.addEventListener("click", () => scoreDataset(btn.dataset.txnId));
  });

  const lookupBtn = document.getElementById("lookup-score-btn");
  if (lookupBtn) {
    lookupBtn.addEventListener("click", () => {
      const id = document.getElementById("txn-id-input").value.trim();
      if (id) scoreDataset(id);
    });
  }

  // --- Case browser: free-form scoring ---
  const customForm = document.getElementById("custom-form");
  const customResult = document.getElementById("custom-result");
  if (customForm) {
    customForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      customResult.classList.remove("hidden");
      customResult.innerHTML = `<p class="muted">Scoring...</p>`;
      const form = new FormData(customForm);
      const resp = await fetch("/api/score/custom", { method: "POST", body: form });
      const data = await resp.json();
      renderScoreResult(customResult, data);
    });
  }

  // --- Mandate demo: live run ---
  const liveRunBtn = document.getElementById("live-run-btn");
  const liveRunStatus = document.getElementById("live-run-status");
  const liveRunResult = document.getElementById("live-run-result");

  function scenarioCardHtml(s, labelPrefix) {
    if (!s.agent_completed) {
      return `<article class="scenario-card"><header><span class="scenario-id">${s.intent_id}</span></header>
        <p class="intent-text">"${s.intent_text}"</p>
        <p class="error">Agent did not complete a purchase for this intent.</p></article>`;
    }
    const p = s.proposed_purchase;
    const demonstrated = s.classifier_missed_mandate_caught;
    let html = `<article class="scenario-card ${demonstrated ? "scenario-demonstrated" : ""}">`;
    html += `<header><span class="scenario-id">${labelPrefix}${s.intent_id}</span>`;
    if (demonstrated) html += `<span class="badge badge-demonstrated">CLASSIFIER MISSED IT -- MANDATE CAUGHT IT</span>`;
    html += `</header>`;
    html += `<p class="intent-text">"${s.intent_text}"</p>`;
    html += `<p class="proposed">Proposed: <strong>${p.product_name}</strong> (${p.category}), $${p.amount.toFixed(2)} ${p.currency}, recurring=${p.recurring}</p>`;
    html += `<div class="checks-row">`;
    html += `<div class="check-box ${s.check_a_merchant.success ? "check-pass" : "check-fail"}"><strong>a. Merchant</strong><br>success=${s.check_a_merchant.success}</div>`;
    html += `<div class="check-box ${s.check_b_classifier.risk_label === "LOW_RISK" ? "check-pass" : "check-fail"}"><strong>b. Defend classifier</strong><br>fraud_probability=${s.check_b_classifier.fraud_probability.toFixed(4)}<br>${s.check_b_classifier.risk_label}</div>`;
    html += `<div class="check-box ${s.check_c_mandate.allowed ? "check-pass" : "check-fail"}"><strong>c. Mandate envelope</strong><br>allowed=${s.check_c_mandate.allowed}</div>`;
    html += `</div>`;
    if (s.refusal_record) {
      html += `<details class="refusal-record"><summary>Refusal record <code>${s.refusal_record.refusal_record_id}</code></summary><ul>`;
      for (const v of s.refusal_record.mandate_binding_violations) {
        html += `<li><strong>${v.field}</strong>: mandate authorized <code>${JSON.stringify(v.mandate_authorized)}</code>, actually proposed <code>${JSON.stringify(v.actually_proposed)}</code> -- ${v.explanation}</li>`;
      }
      html += `</ul></details>`;
    }
    html += `</article>`;
    return html;
  }

  function renderLiveResult(payload) {
    const r = payload.result;
    let html = `<div class="panel" style="margin-top:16px;border-color:var(--good);">`;
    html += `<h3>Live run just completed</h3>`;
    html += `<p class="muted">${r.demonstrated_cases.length} of ${r.scenarios.length} scenarios: classifier LOW_RISK, mandate refused.</p>`;
    html += `<p class="muted">Real OpenAI usage this run: ${r.usage.api_calls} API call(s), `
      + `${r.usage.total_tokens} tokens (${r.usage.prompt_tokens} prompt / ${r.usage.completion_tokens} completion), `
      + `estimated cost $${r.usage.estimated_cost_usd.toFixed(6)} at ${r.usage.model} pricing.</p>`;
    for (const s of r.scenarios) html += scenarioCardHtml(s, "LIVE ");
    html += `<details class="panel-sub"><summary>Verified: every HTTP call this live run made (127.0.0.1 only)</summary><ul class="url-list">`;
    for (const url of r.all_urls_called) html += `<li><code>${url}</code></li>`;
    html += `</ul></details></div>`;
    liveRunResult.innerHTML = html;
  }

  function renderFallback(payload, reasonLabel) {
    let html = `<div class="panel" style="margin-top:16px;border-color:var(--bad);">`;
    html += `<h3>${reasonLabel}</h3>`;
    html += `<p>${payload.reason || ""}</p>`;
    if (payload.fallback) {
      html += `<p class="muted">Falling back to the pre-captured transcript already shown below (<code>${payload.fallback._source_file}</code>) -- unaffected.</p>`;
    }
    html += `</div>`;
    liveRunResult.innerHTML = html;
  }

  async function refreshRateLimitText() {
    const el = document.getElementById("rate-limit-text");
    if (!el) return;
    const resp = await fetch("/api/mandate-demo/rate-limit-status");
    const data = await resp.json();
    el.textContent = `${data.session_runs_remaining} of ${data.max_per_session} runs left this session, max ${data.max_per_minute}/minute globally.`;
    if (liveRunBtn) liveRunBtn.disabled = !data.allowed;
  }

  if (liveRunBtn) {
    liveRunBtn.addEventListener("click", async () => {
      liveRunBtn.disabled = true;
      liveRunStatus.textContent = "Running live agent against the local mock merchant (this calls OpenAI server-side; may take up to ~90s for all 6 scenarios)...";
      liveRunResult.innerHTML = "";
      try {
        const resp = await fetch("/api/mandate-demo/live-run", { method: "POST" });
        const payload = await resp.json();
        if (payload.status === "ok") {
          liveRunStatus.textContent = "Done.";
          renderLiveResult(payload);
        } else if (payload.status === "rate_limited") {
          liveRunStatus.textContent = "Rate limited.";
          renderFallback(payload, "Live run blocked by rate limit");
        } else if (payload.status === "timeout") {
          liveRunStatus.textContent = "Timed out.";
          renderFallback(payload, "Live run timed out");
        } else {
          liveRunStatus.textContent = "Failed.";
          renderFallback(payload, "Live run failed");
        }
      } catch (err) {
        liveRunStatus.textContent = "Failed.";
        renderFallback({ reason: String(err) }, "Live run failed (network error)");
      } finally {
        await refreshRateLimitText();
      }
    });
    refreshRateLimitText();
  }
});
