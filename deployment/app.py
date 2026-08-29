"""FastAPI application exposing the locked ICVS research interface."""

# ruff: noqa: E501

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from .predictor import ICVSPredictor


class ICVSRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age_years: FiniteFloat = Field(ge=18, le=100)
    mgmt_promoter_methylated: bool
    extent_of_resection: Literal["gross_total", "non_gross_total"]
    vit_score_standardized: FiniteFloat


app = FastAPI(title="ICVS-GBM-PFS research API", version="1.1.0")
predictor = ICVSPredictor()


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ICVS-GBM-PFS</title>
  <style>
    :root { --navy:#18324a; --blue:#2f6b8a; --orange:#c64f00; --ink:#1f2933; --muted:#65727c; --line:#d9e0e4; --bg:#f4f7f8; }
    * { box-sizing:border-box; }
    body { margin:0; color:var(--ink); background:var(--bg); font-family:Arial, Helvetica, sans-serif; }
    header { background:var(--navy); color:white; padding:28px 24px; border-top:8px solid #69a7b7; }
    header div { max-width:980px; margin:auto; }
    h1 { margin:0 0 8px; font:700 34px/1.15 Georgia, "Times New Roman", serif; }
    header p { margin:0; color:#d8e7ed; }
    main { max-width:980px; margin:28px auto 50px; padding:0 18px; display:grid; grid-template-columns:minmax(300px,.9fr) minmax(320px,1.1fr); gap:24px; }
    .card { background:white; border:1px solid var(--line); border-radius:14px; padding:24px; box-shadow:0 8px 24px rgba(24,50,74,.07); }
    h2 { margin:0 0 20px; font:700 23px/1.2 Georgia, "Times New Roman", serif; color:var(--navy); }
    label { display:block; font-weight:700; font-size:14px; margin:16px 0 6px; }
    input, select { width:100%; padding:11px 12px; border:1px solid #aebbc3; border-radius:8px; background:white; font-size:15px; color:var(--ink); }
    button { width:100%; margin-top:22px; border:0; border-radius:9px; background:var(--blue); color:white; padding:12px; font-weight:700; font-size:16px; cursor:pointer; }
    button:hover { background:#245775; }
    button:disabled { opacity:.55; cursor:wait; }
    .instructions { color:var(--muted); margin-top:4px; }
    .headline { display:flex; align-items:end; justify-content:space-between; gap:18px; padding-bottom:18px; border-bottom:1px solid var(--line); }
    .risk { font:700 28px/1 Georgia, "Times New Roman", serif; color:var(--orange); }
    .score { text-align:right; color:var(--muted); font-size:13px; }
    .score b { display:block; color:var(--ink); font-size:20px; margin-top:4px; }
    .grid { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-top:20px; }
    .metric { border:1px solid var(--line); border-radius:10px; padding:14px 10px; text-align:center; background:#fbfcfc; }
    .metric span { display:block; font-size:12px; color:var(--muted); margin-bottom:6px; }
    .metric b { color:var(--navy); font-size:20px; }
    .note { margin-top:20px; padding:13px 14px; background:#eef4f6; border-left:4px solid #69a7b7; color:#3e4c55; font-size:13px; line-height:1.45; }
    .attribution { margin-top:22px; }
    .attribution h3 { margin:0 0 10px; color:var(--navy); font-size:16px; }
    table { width:100%; border-collapse:collapse; font-size:12px; }
    th, td { border-bottom:1px solid var(--line); padding:7px 5px; text-align:right; }
    th:first-child, td:first-child { text-align:left; }
    .links { margin-top:18px; font-size:13px; }
    a { color:var(--blue); }
    .error { color:#a12a20; background:#fff1ef; border:1px solid #f1c3bd; padding:12px; border-radius:8px; }
    @media (max-width:760px) { main { grid-template-columns:1fr; } .grid { grid-template-columns:repeat(2,1fr); } }
  </style>
</head>
<body>
  <header><div><h1>Integrated Clinical–ViT Survival Model</h1><p>Progression-free survival estimation for IDH-wildtype glioblastoma</p></div></header>
  <main>
    <section class="card">
      <h2>Model inputs</h2>
      <form id="form">
        <label for="age">Age, years</label><input id="age" type="number" min="18" max="100" step="0.1" required>
        <label for="mgmt">MGMT promoter methylation</label><select id="mgmt" required><option value="" selected disabled>Select status</option><option value="true">Methylated</option><option value="false">Unmethylated</option></select>
        <label for="eor">Extent of resection</label><select id="eor" required><option value="" selected disabled>Select extent</option><option value="gross_total">Gross-total resection</option><option value="non_gross_total">Non-gross-total resection</option></select>
        <label for="vit">Standardized 3D-ViT score</label><input id="vit" type="number" step="any" required>
        <button id="submit" type="submit">Calculate predicted PFS</button>
      </form>
    </section>
    <section class="card" aria-live="polite">
      <h2>Prediction</h2>
      <div id="result" class="instructions">Complete all four inputs to calculate the locked model output.</div>
      <div class="note">Research use only. The standardized 3D-ViT score must be produced by the locked MRI preprocessing and transformer pipeline. This interface is not intended for clinical decision-making.</div>
      <div class="links"><a href="/docs">API documentation</a> · <a href="/health">Service status</a></div>
    </section>
  </main>
  <script>
    const form = document.getElementById('form');
    const result = document.getElementById('result');
    const button = document.getElementById('submit');
    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      button.disabled = true;
      button.textContent = 'Calculating…';
      const payload = {
        age_years:Number(document.getElementById('age').value),
        mgmt_promoter_methylated:document.getElementById('mgmt').value === 'true',
        extent_of_resection:document.getElementById('eor').value,
        vit_score_standardized:Number(document.getElementById('vit').value)
      };
      try {
        const response = await fetch('/predict', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload) });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || 'Prediction failed.');
        const months = [6,12,18,24,30,36];
        const riskLabel = data.risk_group === 'high' ? 'High risk' : 'Low risk';
        result.className = '';
        const labels = { age_years:'Age', mgmt_methylated:'MGMT methylation', non_gross_total_resection:'Non-gross-total resection', vit_score_standardized:'3D-ViT score' };
        const shapleyRows = Object.entries(data.shapley_values).map(([feature, values]) => `<tr><td>${labels[feature]}</td>${months.map(m => `<td>${(100*values[`${m}m`]).toFixed(1)}</td>`).join('')}</tr>`).join('');
        result.innerHTML = `<div class="headline"><div class="risk">${riskLabel}</div><div class="score">Continuous risk score<b>${data.risk_score.toFixed(3)}</b></div></div><div class="grid">${months.map(m => `<div class="metric"><span>${m}-month PFS</span><b>${(100*data[`pfs_probability_${m}m`]).toFixed(1)}%</b></div>`).join('')}</div><div class="attribution"><h3>Contribution to progression risk, percentage points</h3><table><thead><tr><th>Predictor</th>${months.map(m => `<th>${m}m</th>`).join('')}</tr></thead><tbody>${shapleyRows}</tbody></table></div>`;
      } catch (error) {
        result.className = 'error';
        result.textContent = error.message;
      } finally {
        button.disabled = false;
        button.textContent = 'Calculate predicted PFS';
      }
    });
  </script>
</body>
</html>"""


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metadata")
def metadata() -> dict[str, object]:
    return {
        "model": "Integrated Clinical–ViT Survival",
        "features": predictor.feature_order,
        "time_horizons_months": [int(value) for value in predictor.horizons],
        "risk_cutoff": predictor.training_cutoff,
        "interpretation": "Exact four-predictor time-dependent Shapley values",
        "intended_use": "Research use only",
    }


@app.post("/predict")
def predict(request: ICVSRequest) -> dict[str, object]:
    try:
        return predictor.predict(request.model_dump())
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
