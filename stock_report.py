"""HTML report generator for stock-level regime strategy."""

import base64
import io
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


COLORS = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
    "#1abc9c", "#e67e22", "#95a5a6", "#34495e", "#e91e63",
]


def _chart_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _cumulative_chart(strategy_returns: dict[str, pd.Series], title: str) -> str:
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, (name, returns) in enumerate(strategy_returns.items()):
        cumulative = (1 + returns).cumprod()
        ax.plot(cumulative.index, cumulative.values,
                label=name, color=COLORS[i % len(COLORS)], linewidth=1.8)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Cumulative Wealth", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return _chart_base64(fig)


def _drawdown_chart(strategy_returns: dict[str, pd.Series], title: str) -> str:
    fig, ax = plt.subplots(figsize=(14, 4))
    for i, (name, returns) in enumerate(strategy_returns.items()):
        cumulative = (1 + returns).cumprod()
        dd = (cumulative / cumulative.cummax() - 1) * 100
        ax.plot(dd.index, dd.values,
                label=name, color=COLORS[i % len(COLORS)], linewidth=1.2)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Drawdown (%)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return _chart_base64(fig)


def _rolling_ic_chart(rolling_ics: dict[str, pd.Series], market: str, threshold: float) -> str:
    fig, ax = plt.subplots(figsize=(14, 3.5))
    for i, (name, ic) in enumerate(rolling_ics.items()):
        ax.plot(ic.index, ic.values, label=f"{name} Rolling IC",
                color=COLORS[i % len(COLORS)], linewidth=1.5)
    ax.axhline(y=threshold, color="#e74c3c", linestyle="--", linewidth=1, label=f"Threshold ({threshold})")
    ax.axhline(y=0, color="#8b949e", linestyle="-", linewidth=0.5)
    ax.fill_between(ic.index, threshold, ic.values.max() if len(ic) > 0 else 0.1,
                     alpha=0.05, color="#3fb950")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Rolling IC", fontsize=12)
    ax.set_title(f"{market.upper()} — Regime Detection (Rolling IC)", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return _chart_base64(fig)


def _build_summary_table_html(summary_df: pd.DataFrame) -> str:
    header = "<tr><th style='text-align:left'>Strategy</th>"
    header += "".join(f"<th>{col}</th>" for col in summary_df.columns)
    header += "</tr>"

    rows = ""
    for strategy in summary_df.index:
        row = summary_df.loc[strategy]
        cells = ""
        for col in summary_df.columns:
            val = row[col]
            col_vals = summary_df[col].tolist()
            if col in ("R/R", "AR (%)"):
                is_best = val == max(col_vals)
            else:
                is_best = val == min(col_vals)
            cls = ' class="best"' if is_best else ""
            cells += f"<td{cls}>{val}</td>"
        rows += f"<tr><td class='strategy-name'>{strategy}</td>{cells}</tr>\n"

    return f"<table><thead>{header}</thead><tbody>{rows}</tbody></table>"


def generate_stock_report(
    jp_returns: dict[str, pd.Series],
    kr_returns: dict[str, pd.Series],
    jp_summary: pd.DataFrame,
    kr_summary: pd.DataFrame,
    jp_rolling_ics: dict[str, pd.Series],
    kr_rolling_ics: dict[str, pd.Series],
    params: dict,
    output_path: str = "stock_report.html",
) -> None:
    """Generate HTML report comparing JP and KR strategies."""

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Generate charts
    jp_cum_img = _cumulative_chart(jp_returns, "Japan — Cumulative Returns")
    kr_cum_img = _cumulative_chart(kr_returns, "Korea — Cumulative Returns")
    jp_dd_img = _drawdown_chart(jp_returns, "Japan — Drawdown")
    kr_dd_img = _drawdown_chart(kr_returns, "Korea — Drawdown")

    jp_ic_img = _rolling_ic_chart(jp_rolling_ics, "JP", params.get("ic_threshold", 0.03))
    kr_ic_img = _rolling_ic_chart(kr_rolling_ics, "KR", params.get("ic_threshold", 0.03))

    jp_table = _build_summary_table_html(jp_summary)
    kr_table = _build_summary_table_html(kr_summary)

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Stock-Level Regime Lead-Lag Strategy</title>
<style>
  :root {{
    --bg: #0d1117; --card: #161b22; --border: #30363d;
    --text: #c9d1d9; --text-muted: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --red: #f85149;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.6;
    padding: 2rem; max-width: 1200px; margin: 0 auto;
  }}
  h1 {{
    font-size: 1.8rem; font-weight: 700; margin-bottom: 0.3rem;
    background: linear-gradient(135deg, #58a6ff, #bc8cff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }}
  h2 {{
    font-size: 1.2rem; font-weight: 600; margin: 2rem 0 1rem;
    color: var(--text); border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem;
  }}
  .subtitle {{ color: var(--text-muted); font-size: 0.9rem; margin-bottom: 2rem; }}
  .card {{
    background: var(--card); border: 1px solid var(--border);
    border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem;
  }}
  .params {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 0.8rem;
  }}
  .param {{ text-align: center; }}
  .param-value {{ font-size: 1.3rem; font-weight: 700; color: var(--accent); }}
  .param-label {{ font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; }}
  table {{
    width: 100%; border-collapse: collapse; font-size: 0.88rem;
  }}
  th, td {{
    padding: 0.55rem 0.8rem; text-align: right; border-bottom: 1px solid var(--border);
  }}
  th {{ color: var(--text-muted); font-weight: 600; font-size: 0.78rem; text-transform: uppercase; }}
  td.strategy-name {{ text-align: left; font-weight: 600; color: var(--accent); }}
  td.best {{ color: var(--green); font-weight: 700; }}
  img.chart {{ width: 100%; border-radius: 6px; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }}
  @media (max-width: 800px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
  .market-tag {{
    display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px;
    font-size: 0.75rem; font-weight: 700; margin-right: 0.5rem;
  }}
  .tag-jp {{ background: #1a3a5c; color: #58a6ff; }}
  .tag-kr {{ background: #3a1a3c; color: #bc8cff; }}
  .formula {{
    background: #1c2128; border: 1px solid var(--border); border-radius: 6px;
    padding: 1rem 1.2rem; font-family: "SF Mono", "Fira Code", monospace;
    font-size: 0.82rem; color: #e6edf3; overflow-x: auto; margin: 0.8rem 0;
    white-space: pre; line-height: 1.8;
  }}
  .footer {{
    text-align: center; color: var(--text-muted); font-size: 0.75rem;
    margin-top: 2rem; padding-top: 1rem; border-top: 1px solid var(--border);
  }}
</style>
</head>
<body>

<h1>Stock-Level Regime Lead-Lag Strategy</h1>
<p class="subtitle">
  Individual stock lead-lag with Rolling IC regime detection<br>
  US &rarr; Japan &amp; US &rarr; Korea &mdash; Report generated {now}
</p>

<!-- Parameters -->
<div class="card">
  <h2 style="margin-top:0">Parameters</h2>
  <div class="params">
    <div class="param"><div class="param-value">{params.get('lambda', 0.7)}</div><div class="param-label">&lambda; (regularization)</div></div>
    <div class="param"><div class="param-value">{params.get('k', 3)}</div><div class="param-label">K (components)</div></div>
    <div class="param"><div class="param-value">{params.get('window', 60)}</div><div class="param-label">L (window)</div></div>
    <div class="param"><div class="param-value">{params.get('q', 0.3)}</div><div class="param-label">q (quantile)</div></div>
    <div class="param"><div class="param-value">{params.get('ic_window', 40)}</div><div class="param-label">IC Window</div></div>
    <div class="param"><div class="param-value">{params.get('ic_threshold', 0.03)}</div><div class="param-label">IC Threshold</div></div>
  </div>
</div>

<!-- Algorithm -->
<div class="card">
  <h2 style="margin-top:0">Core Algorithm</h2>
  <div class="formula">1. Sector PCA Signal:
   C_reg(t) = (1-&lambda;)C_sample + &lambda;C_prior    (sector-aware regularization)
   signal(t) = V_target &middot; V_us' &middot; z_us(t)         (cross-market factor projection)

2. Sector Cross-Correlation Signal:
   signal_j(t) = mean(r_us_sector(t))            (US sector avg &rarr; target stocks)

3. Regime Detection:
   IC(t) = SpearmanCorr(signal(t-1), return(t))   (daily information coefficient)
   Active(t) = Rolling_IC(t, M={params.get('ic_window', 40)}) > {params.get('ic_threshold', 0.03)}</div>
</div>

<!-- Japan Section -->
<div class="card">
  <h2 style="margin-top:0"><span class="market-tag tag-jp">JP</span>Japan — Performance Summary</h2>
  {jp_table}
</div>

<div class="card">
  <h2 style="margin-top:0"><span class="market-tag tag-jp">JP</span>Cumulative Returns</h2>
  <img class="chart" src="data:image/png;base64,{jp_cum_img}" alt="JP Cumulative">
</div>

<div class="card">
  <h2 style="margin-top:0"><span class="market-tag tag-jp">JP</span>Regime Detection</h2>
  <img class="chart" src="data:image/png;base64,{jp_ic_img}" alt="JP Rolling IC">
</div>

<div class="card">
  <h2 style="margin-top:0"><span class="market-tag tag-jp">JP</span>Drawdown</h2>
  <img class="chart" src="data:image/png;base64,{jp_dd_img}" alt="JP Drawdown">
</div>

<!-- Korea Section -->
<div class="card">
  <h2 style="margin-top:0"><span class="market-tag tag-kr">KR</span>Korea — Performance Summary</h2>
  {kr_table}
</div>

<div class="card">
  <h2 style="margin-top:0"><span class="market-tag tag-kr">KR</span>Cumulative Returns</h2>
  <img class="chart" src="data:image/png;base64,{kr_cum_img}" alt="KR Cumulative">
</div>

<div class="card">
  <h2 style="margin-top:0"><span class="market-tag tag-kr">KR</span>Regime Detection</h2>
  <img class="chart" src="data:image/png;base64,{kr_ic_img}" alt="KR Rolling IC">
</div>

<div class="card">
  <h2 style="margin-top:0"><span class="market-tag tag-kr">KR</span>Drawdown</h2>
  <img class="chart" src="data:image/png;base64,{kr_dd_img}" alt="KR Drawdown">
</div>

<!-- Strategy Descriptions -->
<div class="card">
  <h2 style="margin-top:0">Strategy Descriptions</h2>
  <table>
    <thead><tr><th style="text-align:left">Name</th><th style="text-align:left">Description</th></tr></thead>
    <tbody>
      <tr><td class="strategy-name">*_PCA</td><td style="text-align:left">Sector-regularized PCA signal, no regime filter</td></tr>
      <tr><td class="strategy-name">*_PCA_REGIME</td><td style="text-align:left">PCA signal + Rolling IC regime filter (trade only when IC &gt; threshold)</td></tr>
      <tr><td class="strategy-name">*_XCORR</td><td style="text-align:left">Sector cross-correlation signal (US sector avg), no regime filter</td></tr>
      <tr><td class="strategy-name">*_XCORR_REGIME</td><td style="text-align:left">Cross-correlation signal + Rolling IC regime filter</td></tr>
      <tr><td class="strategy-name">*_MOM</td><td style="text-align:left">Momentum baseline: rolling mean of target close-to-close returns</td></tr>
    </tbody>
  </table>
</div>

<div class="footer">
  Stock-Level Regime-Dependent Lead-Lag Strategy — US &rarr; Japan &amp; US &rarr; Korea
</div>

</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"HTML report saved to: {output_path}")
