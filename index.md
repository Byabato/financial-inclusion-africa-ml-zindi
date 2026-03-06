---
layout: null
---
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Financial Inclusion in Africa — Kelvin Byabato</title>
  <meta name="description" content="End-to-end ML pipeline with SHAP Explainability and Policy Intervention Simulator for East Africa. Stacking Ensembles, Optuna, CatBoost, XGBoost, LightGBM."/>
  <meta property="og:title" content="Financial Inclusion in Africa — Kelvin Byabato"/>
  <meta property="og:image" content="outputs/shap_bar_importance.png"/>
  <link rel="icon" href="https://raw.githubusercontent.com/Byabato/financial-inclusion-africa-ml/main/outputs/02_inclusion_by_country.png" type="image/png">
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --bg:      #0d1117;
      --bg2:     #161c27;
      --bg3:     #1d2535;
      --border:  #263047;
      --text:    #dde4f0;
      --muted:   #7e8fa8;
      --accent:  #3eb58a;
      --blue:    #5b8dee;
      --gold:    #e8b84b;
      --serif:   'DM Serif Display', Georgia, serif;
      --sans:    'DM Sans', system-ui, sans-serif;
      --mono:    'JetBrains Mono', monospace;
      --r:       10px;
    }
    html { scroll-behavior: smooth; }
    body { font-family: var(--sans); background: var(--bg); color: var(--text); line-height: 1.7; overflow-x: hidden; }

    /* NAV */
    nav {
      position: sticky; top: 0; z-index: 100;
      background: rgba(13,17,23,0.93); backdrop-filter: blur(14px);
      border-bottom: 1px solid var(--border);
      padding: 0 2.5rem;
      display: flex; align-items: center; justify-content: space-between;
      height: 56px;
    }
    .nav-brand { font-family: var(--serif); font-size: 1rem; color: var(--text); text-decoration: none; }
    .nav-links { display: flex; gap: 1.8rem; list-style: none; }
    .nav-links a { color: var(--muted); text-decoration: none; font-size: 0.8rem; font-weight: 500; letter-spacing: 0.06em; text-transform: uppercase; transition: color .2s; }
    .nav-links a:hover { color: var(--accent); }
    .nav-cta { background: var(--accent) !important; color: #0d1117 !important; padding: 5px 14px !important; border-radius: 20px !important; font-weight: 600 !important; }
    .nav-cta:hover { opacity: 0.88; }

    /* HERO */
    .hero { max-width: 1100px; margin: 0 auto; padding: 6rem 2.5rem 4.5rem; }
    .eyebrow { font-family: var(--mono); font-size: 0.73rem; color: var(--accent); letter-spacing: .16em; text-transform: uppercase; margin-bottom: 1.2rem; }
    .hero h1 { font-family: var(--serif); font-size: clamp(2.8rem,5.5vw,4.8rem); line-height: 1.08; color: var(--text); margin-bottom: .25rem; }
    .hero h1 em { color: var(--accent); font-style: normal; }
    .hero-sub { font-family: var(--serif); font-style: italic; font-size: clamp(1rem,2vw,1.5rem); color: var(--muted); margin-bottom: 1.8rem; }
    .hero-desc { max-width: 620px; color: var(--muted); font-size: 1rem; line-height: 1.85; margin-bottom: 2rem; }
    .hero-desc strong { color: var(--text); }

    .badges { display: flex; flex-wrap: wrap; gap: .45rem; margin-bottom: 2.2rem; }
    .badge { font-family: var(--mono); font-size: .7rem; padding: 3px 9px; border-radius: 4px; border: 1px solid; letter-spacing: .04em; }
    .b-g { color: var(--accent); border-color: var(--accent); background: rgba(62,181,138,.07); }
    .b-b { color: var(--blue); border-color: var(--blue); background: rgba(91,141,238,.07); }
    .b-y { color: var(--gold); border-color: var(--gold); background: rgba(232,184,75,.07); }
    .b-m { color: var(--muted); border-color: var(--border); }

    .stats { display: flex; gap: 3rem; flex-wrap: wrap; margin-bottom: 2.5rem; }
    .stat-val { display: block; font-family: var(--serif); font-size: 2.3rem; color: var(--accent); line-height: 1; }
    .stat-lbl { font-family: var(--mono); font-size: .68rem; color: var(--muted); text-transform: uppercase; letter-spacing: .1em; margin-top: 3px; }

    .btns { display: flex; gap: .9rem; flex-wrap: wrap; margin-bottom: 3rem; }
    .btn { display: inline-flex; align-items: center; gap: 6px; padding: 10px 22px; border-radius: var(--r); font-size: .87rem; font-weight: 600; text-decoration: none; transition: all .18s; }
    .btn-p { background: var(--accent); color: #0d1117; border: 1px solid var(--accent); }
    .btn-p:hover { background: #4ecf9f; transform: translateY(-1px); }
    .btn-g { background: transparent; color: var(--text); border: 1px solid var(--border); }
    .btn-g:hover { border-color: var(--accent); color: var(--accent); transform: translateY(-1px); }

    .socials { display: flex; gap: .8rem; flex-wrap: wrap; }
    .soc { display: inline-flex; align-items: center; gap: 6px; color: var(--muted); text-decoration: none; font-size: .8rem; font-weight: 500; padding: 6px 12px; border: 1px solid var(--border); border-radius: 6px; transition: all .18s; }
    .soc:hover { color: var(--accent); border-color: var(--accent); }
    .soc svg { width: 15px; height: 15px; fill: currentColor; flex-shrink: 0; }

    /* SECTIONS */
    .sec { max-width: 1100px; margin: 0 auto; padding: 5rem 2.5rem; }
    hr.div { border: none; border-top: 1px solid var(--border); margin: 0 2.5rem; }
    .sec-lbl { font-family: var(--mono); font-size: .7rem; color: var(--accent); letter-spacing: .15em; text-transform: uppercase; margin-bottom: .5rem; }
    .sec-title { font-family: var(--serif); font-size: clamp(1.7rem,3vw,2.5rem); color: var(--text); line-height: 1.15; margin-bottom: .9rem; }
    .sec-body { color: var(--muted); max-width: 660px; line-height: 1.85; margin-bottom: 2.2rem; font-size: .98rem; }
    .sec-body strong { color: var(--text); }

    /* CARDS */
    .card { background: var(--bg2); border: 1px solid var(--border); border-radius: var(--r); overflow: hidden; }
    .card img { width: 100%; display: block; }
    .cap { padding: 9px 13px; font-family: var(--mono); font-size: .69rem; color: var(--muted); border-top: 1px solid var(--border); }

    /* GRIDS */
    .g2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1.2rem; }
    .g3 { display: grid; grid-template-columns: repeat(3,1fr); gap: 1.4rem; }
    .g4 { display: grid; grid-template-columns: repeat(4,1fr); gap: 1rem; }
    @media (max-width: 860px) { .g3 { grid-template-columns: 1fr; } .g4 { grid-template-columns: 1fr 1fr; } }
    @media (max-width: 640px) { .g2, .g4 { grid-template-columns: 1fr; } }

    /* LAYER CARDS */
    .layer { background: var(--bg2); border: 1px solid var(--border); border-radius: var(--r); padding: 2rem 1.6rem; position: relative; transition: border-color .2s, transform .2s; }
    .layer:hover { border-color: var(--accent); transform: translateY(-3px); }
    .layer-n { font-family: var(--serif); font-size: 3.5rem; color: var(--bg3); position: absolute; top: .8rem; right: 1.2rem; line-height: 1; }
    .layer h3 { font-family: var(--mono); font-size: .68rem; color: var(--accent); text-transform: uppercase; letter-spacing: .12em; margin-bottom: .4rem; }
    .layer h4 { font-family: var(--serif); font-size: 1.35rem; color: var(--text); margin-bottom: .7rem; }
    .layer p { color: var(--muted); font-size: .9rem; line-height: 1.72; }
    .layer-m { margin-top: 1.1rem; font-family: var(--mono); font-size: .75rem; color: var(--accent); background: rgba(62,181,138,.08); border: 1px solid rgba(62,181,138,.2); padding: 5px 11px; border-radius: 5px; display: inline-block; }

    /* PIPELINE */
    .pipe { background: var(--bg2); border: 1px solid var(--border); border-radius: var(--r); padding: 1.8rem 2rem; margin-top: 1.5rem; }
    .pipe-step { display: flex; align-items: flex-start; gap: 1.1rem; padding: .9rem 0; border-bottom: 1px solid var(--border); }
    .pipe-step:last-child { border-bottom: none; }
    .step-n { width: 26px; height: 26px; border-radius: 50%; background: var(--bg3); border: 1px solid var(--accent); color: var(--accent); font-family: var(--mono); font-size: .66rem; display: flex; align-items: center; justify-content: center; flex-shrink: 0; margin-top: 2px; }
    .step-c h4 { color: var(--text); font-size: .93rem; font-weight: 600; margin-bottom: 3px; }
    .step-c p { color: var(--muted); font-size: .84rem; line-height: 1.65; }

    /* TABLES */
    table { width: 100%; border-collapse: collapse; font-size: .88rem; margin-top: 1.5rem; }
    th { background: var(--bg3); color: var(--muted); font-family: var(--mono); font-size: .67rem; font-weight: 500; text-transform: uppercase; letter-spacing: .08em; padding: 9px 13px; text-align: left; border-bottom: 1px solid var(--border); }
    td { padding: 10px 13px; border-bottom: 1px solid var(--border); color: var(--muted); }
    td:first-child { color: var(--text); }
    .row-best td { color: var(--text); font-weight: 600; background: rgba(62,181,138,.05); }
    .row-best td:nth-child(2), .row-best td:nth-child(3) { color: var(--accent); }
    .delta { color: var(--accent); font-family: var(--mono); font-size: .78rem; }
    .rate { color: var(--accent); font-weight: 700; font-family: var(--mono); }
    td:last-child { color: var(--blue); font-family: var(--mono); font-size: .8rem; }

    /* INSIGHT CARDS */
    .ins-g { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1.5rem; }
    @media (max-width: 640px) { .ins-g { grid-template-columns: 1fr; } }
    .ins { background: var(--bg2); border: 1px solid var(--border); border-left: 3px solid var(--accent); border-radius: var(--r); padding: 1.2rem 1.3rem; }
    .ins-r { font-family: var(--mono); font-size: .66rem; color: var(--accent); text-transform: uppercase; letter-spacing: .1em; margin-bottom: .25rem; }
    .ins h4 { color: var(--text); font-size: .93rem; font-weight: 600; margin-bottom: .35rem; }
    .ins p { color: var(--muted); font-size: .84rem; line-height: 1.65; }

    /* ACCORDION */
    .acc { margin-top: 1.5rem; display: flex; flex-direction: column; gap: .55rem; }
    details { background: var(--bg2); border: 1px solid var(--border); border-radius: var(--r); overflow: hidden; }
    details[open] { border-color: var(--accent); }
    summary { padding: .95rem 1.3rem; cursor: pointer; font-weight: 600; color: var(--text); font-size: .92rem; list-style: none; display: flex; align-items: center; justify-content: space-between; user-select: none; }
    summary::-webkit-details-marker { display: none; }
    summary::after { content: '+'; font-family: var(--mono); font-size: 1.2rem; color: var(--accent); }
    details[open] summary::after { content: '−'; }
    .det { padding: 0 1.3rem 1.1rem; color: var(--muted); font-size: .88rem; line-height: 1.8; border-top: 1px solid var(--border); padding-top: .9rem; }
    .det strong { color: var(--text); }

    /* CODE */
    .code { background: var(--bg3); border: 1px solid var(--border); border-radius: var(--r); padding: 1.3rem 1.5rem; font-family: var(--mono); font-size: .78rem; color: #a0aec0; line-height: 1.9; overflow-x: auto; margin-top: 1rem; white-space: pre; }
    .cg { color: var(--accent); } .cb { color: var(--blue); } .cy { color: var(--gold); } .cm { color: #394558; }

    /* CONTACT */
    .con { background: var(--bg2); border: 1px solid var(--border); border-radius: var(--r); padding: 1.3rem; text-decoration: none; transition: all .18s; display: flex; flex-direction: column; gap: .45rem; }
    .con:hover { border-color: var(--accent); transform: translateY(-2px); }
    .con-p { font-family: var(--mono); font-size: .66rem; color: var(--accent); text-transform: uppercase; letter-spacing: .1em; }
    .con-n { color: var(--text); font-weight: 600; font-size: .93rem; }
    .con-d { color: var(--muted); font-size: .79rem; line-height: 1.5; }

    /* FOOTER */
    footer { border-top: 1px solid var(--border); padding: 2rem 2.5rem; text-align: center; color: var(--muted); font-size: .8rem; line-height: 1.9; }
    footer a { color: var(--accent); text-decoration: none; }
    footer a:hover { text-decoration: underline; }

    /* ANIMATIONS */
    @keyframes up { from { opacity:0; transform:translateY(18px); } to { opacity:1; transform:translateY(0); } }
    .hero > * { animation: up .55s ease both; }
    .hero > *:nth-child(1){animation-delay:.04s} .hero > *:nth-child(2){animation-delay:.1s}
    .hero > *:nth-child(3){animation-delay:.16s} .hero > *:nth-child(4){animation-delay:.22s}
    .hero > *:nth-child(5){animation-delay:.28s} .hero > *:nth-child(6){animation-delay:.34s}
    .hero > *:nth-child(7){animation-delay:.4s}  .hero > *:nth-child(8){animation-delay:.46s}

    ::-webkit-scrollbar { width: 5px; } ::-webkit-scrollbar-track { background: var(--bg); } ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

    @media (max-width: 640px) { nav { padding: 0 1.2rem; } .nav-links { display: none; } .sec { padding: 3.5rem 1.3rem; } .hero { padding: 4rem 1.3rem 3rem; } .stats { gap: 1.6rem; } }

    code { font-family: var(--mono); font-size: .8rem; background: var(--bg3); padding: 1px 5px; border-radius: 3px; color: var(--gold); }
  </style>
</head>
<body>

<nav>
  <a href="#" class="nav-brand">Kelvin Byabato</a>
  <ul class="nav-links">
    <li><a href="#problem">Problem</a></li>
    <li><a href="#eda">Analysis</a></li>
    <li><a href="#model">Model</a></li>
    <li><a href="#xai">Explainability</a></li>
    <li><a href="#simulator">Simulator</a></li>
    <li><a href="#contact" class="nav-cta">Contact</a></li>
  </ul>
</nav>

<!-- HERO -->
<div class="hero">
  <div class="eyebrow">Data Science · East Africa · 2024</div>
  <h1>Financial Inclusion<br><em>in Africa</em></h1>
  <div class="hero-sub">Predictive ML · Explainable AI · Policy Simulation</div>
  <p class="hero-desc">An end-to-end machine learning pipeline that predicts financial exclusion across Kenya, Rwanda, Tanzania and Uganda — then explains <strong>why</strong> individuals are excluded and recommends <strong>SDG-aligned interventions</strong>.</p>

  <div class="badges">
    <span class="badge b-g">OOF MAE 0.1117</span>
    <span class="badge b-g">AUC 0.8647</span>
    <span class="badge b-b">Stacking Ensemble</span>
    <span class="badge b-b">Optuna · 50 trials</span>
    <span class="badge b-y">SHAP TreeExplainer</span>
    <span class="badge b-m">XGBoost · LightGBM · CatBoost</span>
    <span class="badge b-m">Lightning AI Studios</span>
  </div>

  <div class="stats">
    <div><span class="stat-val">0.1117</span><span class="stat-lbl">OOF MAE</span></div>
    <div><span class="stat-val">0.8647</span><span class="stat-lbl">ROC-AUC</span></div>
    <div><span class="stat-val">33.6k</span><span class="stat-lbl">Respondents</span></div>
    <div><span class="stat-val">20+</span><span class="stat-lbl">Features</span></div>
    <div><span class="stat-val">4</span><span class="stat-lbl">Countries</span></div>
  </div>

  <div class="btns">
    <a href="https://github.com/byabato/financial-inclusion-africa-ml-zindi" class="btn btn-p" target="_blank">View Source Code</a>
    <a href="#simulator" class="btn btn-g">See Policy Simulator</a>
    <a href="#contact" class="btn btn-g">Connect</a>
  </div>

  <div class="socials">
    <a href="https://www.linkedin.com/in/kelvin-byabato" class="soc" target="_blank">
      <svg viewBox="0 0 24 24"><path d="M19 0H5C2.239 0 0 2.239 0 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5V5c0-2.761-2.238-5-5-5zM8 19H5V8h3v11zM6.5 6.732c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zM20 19h-3v-5.604c0-3.368-4-3.113-4 0V19h-3V8h3v1.765C14.396 7.179 20 6.988 20 12.24V19z"/></svg>
      LinkedIn
    </a>
    <a href="https://zindi.africa/users/kelvin_byb" class="soc" target="_blank">
      <svg viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2V8h2v9zm0-11h-2V4h2v2z"/></svg>
      Zindi · kelvin_byb
    </a>
    <a href="https://www.instagram.com/kelvin_byb/" class="soc" target="_blank">
      <svg viewBox="0 0 24 24"><path d="M12 2.163c3.204 0 3.584.012 4.85.07 3.252.148 4.771 1.691 4.919 4.919.058 1.265.069 1.645.069 4.849 0 3.205-.012 3.584-.069 4.849-.149 3.225-1.664 4.771-4.919 4.919-1.266.058-1.644.07-4.85.07-3.204 0-3.584-.012-4.849-.07-3.26-.149-4.771-1.699-4.919-4.92-.058-1.265-.07-1.644-.07-4.849 0-3.204.013-3.583.07-4.849.149-3.227 1.664-4.771 4.919-4.919 1.266-.057 1.645-.069 4.849-.069zM12 0C8.741 0 8.333.014 7.053.072 2.695.272.273 2.69.073 7.052.014 8.333 0 8.741 0 12c0 3.259.014 3.668.072 4.948.2 4.358 2.618 6.78 6.98 6.98C8.333 23.986 8.741 24 12 24c3.259 0 3.668-.014 4.948-.072 4.354-.2 6.782-2.618 6.979-6.98.059-1.28.073-1.689.073-4.948 0-3.259-.014-3.667-.072-4.947-.196-4.354-2.617-6.78-6.979-6.98C15.668.014 15.259 0 12 0zm0 5.838a6.162 6.162 0 100 12.324 6.162 6.162 0 000-12.324zM12 16a4 4 0 110-8 4 4 0 010 8zm6.406-11.845a1.44 1.44 0 100 2.881 1.44 1.44 0 000-2.881z"/></svg>
      @kelvin_byb
    </a>
  </div>
</div>

<hr class="div"/>

<!-- PROBLEM -->
<section class="sec" id="problem">
  <div class="sec-lbl">Context</div>
  <h2 class="sec-title">The Problem That Started This</h2>

  <div class="g2" style="align-items:start; margin-bottom:3rem">
    <div>
      <p style="color:var(--text); font-family:var(--serif); font-size:1.4rem; line-height:1.3; margin-bottom:1rem">86% of East Africans have no bank account.</p>
      <p style="color:var(--muted); line-height:1.85; margin-bottom:1rem; font-size:.97rem">This is not a personal failure. It is a structural one — shaped by employment type, education level, gender, geography, and whether someone owns a mobile phone. Without data-driven evidence, policymakers allocate resources blindly.</p>
      <p style="color:var(--muted); line-height:1.85; margin-bottom:1rem; font-size:.97rem">This project answers three questions that actually matter:</p>
      <div style="display:flex; flex-direction:column; gap:.6rem; margin-bottom:1.2rem">
        <div style="display:flex; gap:.8rem; align-items:flex-start">
          <span style="font-family:var(--mono); font-size:.75rem; color:var(--accent); background:rgba(62,181,138,.1); border:1px solid rgba(62,181,138,.2); padding:3px 8px; border-radius:4px; flex-shrink:0; margin-top:2px">WHO</span>
          <span style="color:var(--muted); font-size:.93rem">Is most likely to be financially excluded?</span>
        </div>
        <div style="display:flex; gap:.8rem; align-items:flex-start">
          <span style="font-family:var(--mono); font-size:.75rem; color:var(--blue); background:rgba(91,141,238,.1); border:1px solid rgba(91,141,238,.2); padding:3px 8px; border-radius:4px; flex-shrink:0; margin-top:2px">WHY</span>
          <span style="color:var(--muted); font-size:.93rem">Are they excluded? Not just a label — a reason with evidence.</span>
        </div>
        <div style="display:flex; gap:.8rem; align-items:flex-start">
          <span style="font-family:var(--mono); font-size:.75rem; color:var(--gold); background:rgba(232,184,75,.1); border:1px solid rgba(232,184,75,.2); padding:3px 8px; border-radius:4px; flex-shrink:0; margin-top:2px">WHAT</span>
          <span style="color:var(--muted); font-size:.93rem">Should be done? Actionable, SDG-aligned interventions.</span>
        </div>
      </div>
      <p style="color:var(--muted); font-size:.85rem">Data: FinAccess + FinScope surveys · 33,600 respondents · 2016–2018 · 14% banked · 6:1 class imbalance</p>
    </div>
    <div class="card">
      <img src="outputs/01_class_distribution.png" alt="Class Distribution"/>
      <div class="cap">Only 14% of 33,600 surveyed adults hold a commercial bank account</div>
    </div>
  </div>

  <div class="g3">
    <div class="layer">
      <span class="layer-n">1</span>
      <h3>Layer One</h3>
      <h4>Predict</h4>
      <p>Stacking ensemble of XGBoost, LightGBM and CatBoost, tuned with Optuna Bayesian search (50 trials each). Stratified K-Fold with MAE-optimised threshold.</p>
      <span class="layer-m">OOF MAE 0.1117 · AUC 0.8647</span>
    </div>
    <div class="layer">
      <span class="layer-n">2</span>
      <h3>Layer Two</h3>
      <h4>Explain</h4>
      <p>SHAP TreeExplainer decodes every prediction. Not a black box — each label carries a ranked list of barriers specific to that individual. Global and local attribution.</p>
      <span class="layer-m">Feature-level causal attribution</span>
    </div>
    <div class="layer">
      <span class="layer-n">3</span>
      <h3>Layer Three</h3>
      <h4>Act</h4>
      <p>A Financial Inclusion Recommender maps each person's top SHAP barriers to concrete interventions. Policymakers receive names, numbers and actions.</p>
      <span class="layer-m">SDG 1 · 4 · 8 · 9 · 10</span>
    </div>
  </div>
</section>

<hr class="div"/>

<!-- EDA -->
<section class="sec" id="eda">
  <div class="sec-lbl">Exploratory Analysis</div>
  <h2 class="sec-title">What the Data Reveals</h2>
  <p class="sec-body">Every chart answers a specific business question. The structural barriers to banking are visible long before any model is trained.</p>

  <div class="g2">
    <div class="card"><img src="outputs/02_inclusion_by_country.png" alt="Country"/><div class="cap">Country gap: Kenya leads at 73% · Uganda and Tanzania below 35%</div></div>
    <div class="card"><img src="outputs/06_gender_gap.png" alt="Gender"/><div class="cap">Gender gap: Men are 5–10 percentage points more banked in all 4 countries</div></div>
    <div class="card"><img src="outputs/05_mobile_vs_banking.png" alt="Mobile"/><div class="cap">Mobile bridge: Cellphone access delivers 3–4× higher banking rate</div></div>
    <div class="card"><img src="outputs/08_urban_rural.png" alt="Urban Rural"/><div class="cap">Urban premium: Urban residents are 2–3× more likely to hold an account</div></div>
    <div class="card"><img src="outputs/04_inclusion_by_education.png" alt="Education"/><div class="cap">Education ladder: Tertiary = 5× higher rate vs. no formal schooling</div></div>
    <div class="card"><img src="outputs/03_inclusion_by_jobtype.png" alt="Employment"/><div class="cap">Employment type: Formal work unlocks banking · No income locks it out</div></div>
  </div>
  <div class="card" style="margin-top:1.2rem"><img src="outputs/10_country_education_heatmap.png" alt="Heatmap"/><div class="cap">Cross-factor heatmap: Country × Education — dark zones mark highest exclusion risk</div></div>
</section>

<hr class="div"/>

<!-- MODEL -->
<section class="sec" id="model">
  <div class="sec-lbl">Machine Learning</div>
  <h2 class="sec-title">Model Architecture and Training</h2>
  <p class="sec-body">Built for production, not just a leaderboard. Every step is justified, every score is out-of-fold — never evaluated on training data.</p>

  <div class="pipe">
    <div class="pipe-step">
      <div class="step-n">01</div>
      <div class="step-c"><h4>Feature Engineering — 20+ features</h4><p>Domain-ordered ordinal encoding (<code>education_rank</code>, <code>employment_rank</code>), composite inclusion score, age lifecycle bins, 7 interaction features, K-Fold smoothed target encoding with zero leakage.</p></div>
    </div>
    <div class="pipe-step">
      <div class="step-n">02</div>
      <div class="step-c"><h4>Stratified K-Fold (k=5)</h4><p>With 14% positive class, random splits create unreliable folds. Stratified splitting guarantees every fold mirrors the full class ratio, making OOF scores trustworthy estimates of leaderboard performance.</p></div>
    </div>
    <div class="pipe-step">
      <div class="step-n">03</div>
      <div class="step-c"><h4>Optuna Bayesian Hyperparameter Search</h4><p>50 trials per model using Tree-structured Parzen Estimation. GridSearch is O(n^k) — Optuna learns which hyperparameter regions are promising. 500 total CV evaluations across XGBoost and LightGBM.</p></div>
    </div>
    <div class="pipe-step">
      <div class="step-n">04</div>
      <div class="step-c"><h4>Stacking Ensemble</h4><p>OOF probability arrays from XGBoost, LightGBM and CatBoost feed a Logistic Regression meta-learner. The meta-model learns optimal trust weights per base model per region of feature space.</p></div>
    </div>
    <div class="pipe-step">
      <div class="step-n">05</div>
      <div class="step-c"><h4>Threshold Optimisation for MAE</h4><p>Zindi evaluates hard 0/1 labels. The default 0.5 threshold is wrong for 6:1 imbalance. A full scan from 0.05 to 0.95 finds the MAE-minimising cutoff — found at 0.88 for this dataset.</p></div>
    </div>
  </div>

  <div class="card" style="margin-top:1.4rem"><img src="outputs/12_model_comparison.png" alt="Model Comparison"/><div class="cap">OOF MAE per model — lower is better. Every model beats the baseline. Stacking wins.</div></div>

  <table>
    <thead><tr><th>Model</th><th>OOF MAE</th><th>OOF AUC</th><th>vs. Baseline</th></tr></thead>
    <tbody>
      <tr><td>Logistic Regression</td><td>~0.170</td><td>~0.750</td><td class="delta">— baseline —</td></tr>
      <tr><td>XGBoost</td><td>~0.130</td><td>~0.840</td><td class="delta">+24%</td></tr>
      <tr><td>LightGBM</td><td>~0.130</td><td>~0.840</td><td class="delta">+24%</td></tr>
      <tr><td>CatBoost</td><td>~0.130</td><td>~0.840</td><td class="delta">+24%</td></tr>
      <tr class="row-best"><td>Stacking Ensemble</td><td>0.1117</td><td>0.8647</td><td class="delta">+34%</td></tr>
    </tbody>
  </table>

  <div class="acc">
    <details>
      <summary>Optuna hyperparameter search space and strategy</summary>
      <div class="det">
        Search space per model: <code>max_depth</code> 3–9 · <code>learning_rate</code> 0.01–0.20 (log-uniform) · <code>n_estimators</code> 200–1000 · <code>subsample</code> 0.5–1.0 · <code>colsample_bytree</code> 0.5–1.0 · <code>reg_alpha</code> and <code>reg_lambda</code> 1e-8–10.0 (log-uniform) · <code>scale_pos_weight</code> 1.0–8.0 to handle 6:1 class imbalance.<br/><br/>50 trials × 2 models × 5-fold CV = 500 full model evaluations. TPE sampler learns the promising regions after each trial.
      </div>
    </details>
  </div>
</section>

<hr class="div"/>

<!-- XAI -->
<section class="sec" id="xai">
  <div class="sec-lbl">Explainable AI</div>
  <h2 class="sec-title">Why Each Person is Predicted Excluded</h2>
  <p class="sec-body">A model that outputs 0 or 1 is a black box. A model that tells you <strong>why</strong> is a tool for change. SHAP TreeExplainer computes exact Shapley values — each feature's marginal contribution to each prediction, grounded in cooperative game theory.</p>

  <div class="g2">
    <div class="card"><img src="outputs/shap_bar_importance.png" alt="SHAP Bar"/><div class="cap">Global importance: mean absolute SHAP value — which features move predictions most</div></div>
    <div class="card"><img src="outputs/shap_summary_beeswarm.png" alt="SHAP Beeswarm"/><div class="cap">Individual impact: each dot is one person · red = high feature value · blue = low</div></div>
  </div>

  <div class="ins-g">
    <div class="ins"><div class="ins-r">Rank 1 · mean |SHAP| 0.426</div><h4><code>cellphone_access</code></h4><p>The single most powerful predictor. No mobile phone means no pathway to mobile money — the primary on-ramp to formal banking in East Africa.</p></div>
    <div class="ins"><div class="ins-r">Rank 2 · mean |SHAP| 0.378</div><h4><code>country_Tanzania</code></h4><p>Being in Tanzania is a significant barrier independent of individual characteristics. Infrastructure gap, not personal failing.</p></div>
    <div class="ins"><div class="ins-r">Rank 3 · mean |SHAP| 0.312</div><h4><code>inclusion_score</code></h4><p>The composite feature captures systemic multi-barrier exclusion — when employment, education, mobile and urban factors all point low simultaneously.</p></div>
    <div class="ins"><div class="ins-r">Rank 4 · mean |SHAP| 0.306</div><h4><code>age_x_education</code></h4><p>Young AND uneducated is a compounding interaction. Neither age nor education alone captures this exclusion risk as strongly as their product.</p></div>
  </div>

  <div class="acc">
    <details>
      <summary>Sample prediction explanation — uniqueid_6065 x Kenya (predicted unbanked)</summary>
      <div class="det">
        <div class="code"><span class="cm"># Top BARRIERS pushing toward unbanked:</span>
<span class="cb">cellphone_access</span>   SHAP = <span class="cy">-0.843</span>   value = 0  <span class="cm">(no phone)</span>
<span class="cb">inclusion_score</span>    SHAP = <span class="cy">-0.617</span>   value = 0.30
<span class="cb">age_x_education</span>    SHAP = <span class="cy">-0.350</span>   value = 0.00
<span class="cb">edu_x_mobile</span>       SHAP = <span class="cy">-0.325</span>   value = 0.00

<span class="cm"># Top ENABLERS pushing toward banked:</span>
<span class="cb">age_of_respondent</span>  SHAP = <span class="cg">+0.496</span>   value = 77
<span class="cb">country_Rwanda</span>     SHAP = <span class="cg">+0.268</span>   value = 0</div>
        <br/>The model does not just output "unbanked." It says: <strong>give this person mobile access first</strong>. The recommender converts this into an SDG 9 intervention automatically.
      </div>
    </details>
  </div>
</section>

<hr class="div"/>

<!-- SIMULATOR -->
<section class="sec" id="simulator">
  <div class="sec-lbl">Innovation Layer</div>
  <h2 class="sec-title">Policy Intervention Simulator</h2>
  <p class="sec-body">For every predicted-unbanked individual, SHAP values are mapped to concrete SDG-aligned recommendations. Policymakers receive a country scorecard — not a confusion matrix.</p>

  <div class="card"><img src="outputs/intervention_simulator.png" alt="Simulator"/><div class="cap">"What if all rural residents had mobile access?" — estimated 8 percentage-point uplift per country</div></div>

  <table style="margin-top:1.5rem">
    <thead><tr><th>Country</th><th>Predicted Inclusion</th><th>Primary Barrier</th><th>Priority Intervention</th><th>SDG</th></tr></thead>
    <tbody>
      <tr><td>Uganda</td><td><span class="rate">32.1%</span></td><td style="color:var(--muted)">Infrastructure gap</td><td style="color:var(--muted)">Agent banking expansion</td><td>SDG 9</td></tr>
      <tr><td>Tanzania</td><td><span class="rate">33.4%</span></td><td style="color:var(--muted)">Cellphone access</td><td style="color:var(--muted)">Mobile money programs</td><td>SDG 9</td></tr>
      <tr><td>Rwanda</td><td><span class="rate">50.8%</span></td><td style="color:var(--muted)">Composite exclusion</td><td style="color:var(--muted)">Multi-factor programs</td><td>SDG 1, 8</td></tr>
      <tr><td>Kenya</td><td><span class="rate">73.2%</span></td><td style="color:var(--muted)">Age-education gap</td><td style="color:var(--muted)">Youth financial literacy</td><td>SDG 4</td></tr>
    </tbody>
  </table>

  <div class="g2" style="margin-top:1.2rem">
    <div class="card"><img src="outputs/14_barrier_distribution.png" alt="Barriers"/><div class="cap">Most common barriers across 9,339 predicted-unbanked individuals</div></div>
    <div class="card"><img src="outputs/shap_dependence_cellphone_access.png" alt="Dependence"/><div class="cap">SHAP dependence: how cellphone access value maps to prediction impact</div></div>
  </div>

  <div class="acc">
    <details>
      <summary>The M-Pesa lesson — why Kenya leads at 73.2%</summary>
      <div class="det">Kenya's inclusion rate is not accidental. It is the direct result of mobile money infrastructure built over two decades. M-Pesa launched in 2007 and gave millions a financial identity before they ever walked into a bank branch.<br/><br/>The model's <strong>number one SHAP feature is cellphone access</strong> — confirming what policy history already showed. The recommendation for Tanzania and Uganda is clear: mobile infrastructure investment comes before branch expansion, literacy programmes or regulatory reform.</div>
    </details>
  </div>
</section>

<hr class="div"/>

<!-- LESSONS -->
<section class="sec" id="lessons">
  <div class="sec-lbl">Engineering Depth</div>
  <h2 class="sec-title">Lessons from Building This</h2>
  <p class="sec-body">The decisions that separate a reliable pipeline from an overfit notebook.</p>
  <div class="acc">
    <details>
      <summary>Data leakage — the silent killer of validation scores</summary>
      <div class="det">Standard target encoding computes group means from the full training set, then uses those means as features. This leaks future information — validation scores look better than they are, but leaderboard scores are much worse. <strong>K-Fold target encoding</strong> computes means from held-out folds only. Each row is encoded using only rows that were not used to train that fold's model. The code difference is 10 lines. The impact on reliability is enormous.</div>
    </details>
    <details>
      <summary>Why MAE requires threshold optimisation, not just model tuning</summary>
      <div class="det">Zindi evaluates hard <code>0/1</code> labels, not probabilities. With 86% of data being class 0, the default 0.5 threshold is systematically biased toward predicting "unbanked" even when the model is uncertain. Scanning thresholds from 0.05 to 0.95 and selecting the MAE-minimising value is not a trick — it is the correct evaluation strategy. The optimal threshold was <strong>0.88</strong>, which reflects the model's correct response to 6:1 imbalance.</div>
    </details>
    <details>
      <summary>SHAP vs. built-in feature importance — they measure different things</summary>
      <div class="det">XGBoost's built-in importance measures how often a feature appears in tree splits. SHAP measures the <strong>actual contribution</strong> of each feature to each prediction. A feature can appear frequently in splits but contribute little to the final score if those splits are near the root on weak signals. SHAP is the honest measure and the one policymakers can act on.</div>
    </details>
    <details>
      <summary>Stratified K-Fold — why random splitting fails on imbalanced data</summary>
      <div class="det">With only 14% positive class, random splits create folds where some have 8% positives and others 20%. The model trains on a different problem in each fold and OOF scores become unreliable. <strong>Stratified K-Fold</strong> guarantees every fold mirrors the 14:86 ratio exactly, making cross-validation scores trustworthy estimates of leaderboard performance.</div>
    </details>
  </div>
</section>

<hr class="div"/>

<!-- CONTACT -->
<section class="sec" id="contact">
  <div class="sec-lbl">Connect</div>
  <h2 class="sec-title">Kelvin Byabato</h2>
  <p class="sec-body">Data scientist building machine learning solutions for real-world problems in East Africa. Open to collaborations, research discussions and opportunities.</p>
  <div class="g4">
    <a href="https://www.linkedin.com/in/kelvin-byabato" class="con" target="_blank"><span class="con-p">LinkedIn</span><span class="con-n">kelvin-byabato</span><span class="con-d">Professional case studies and networking</span></a>
    <a href="https://zindi.africa/users/kelvin_byb" class="con" target="_blank"><span class="con-p">Zindi Africa</span><span class="con-n">kelvin_byb</span><span class="con-d">Competition history and hackathons</span></a>
    <a href="https://www.instagram.com/kelvin_byb/" class="con" target="_blank"><span class="con-p">Instagram</span><span class="con-n">@kelvin_byb</span><span class="con-d">Tech community and daily updates</span></a>
    <a href="https://github.com/byabato/financial-inclusion-africa-ml-zindi" class="con" target="_blank"><span class="con-p">GitHub</span><span class="con-n">Source Code</span><span class="con-d">Full pipeline, notebooks and documentation</span></a>
  </div>
  <div style="margin-top:2.8rem; text-align:center">
    <a href="https://github.com/byabato/financial-inclusion-africa-ml-zindi" class="btn btn-p" target="_blank" style="font-size:.95rem; padding:13px 30px">View Full Source Code on GitHub</a>
  </div>
</section>

<footer>
  <p>Built by <strong>Kelvin Byabato</strong> on Lightning AI Studios · Deployed on GitHub Pages</p>
  <p style="margin-top:.3rem"><a href="https://zindi.africa/competitions/financial-inclusion-in-africa" target="_blank">Zindi Challenge</a> · <a href="https://arxiv.org/abs/1705.07874" target="_blank">SHAP Paper</a> · <a href="https://data.worldbank.org/indicator/FX.OWN.TOTL.ZS" target="_blank">World Bank Findex</a></p>
  <p style="margin-top:.3rem; color:#394558; font-size:.76rem">For the African Data Science Community — because 86% of East Africans deserve better.</p>
</footer>

</body>
</html>
