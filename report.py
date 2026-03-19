"""HTML report generator for strategy results."""

import base64
import io
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _cumulative_chart_base64(strategy_returns: dict[str, pd.Series]) -> str:
    """Render cumulative return chart and return as base64-encoded PNG."""
    colors = {
        "PCA_SUB": "#e74c3c",
        "DOUBLE": "#3498db",
        "PCA_PLAIN": "#2ecc71",
        "MOM": "#95a5a6",
    }

    fig, ax = plt.subplots(figsize=(14, 5))
    for name, returns in strategy_returns.items():
        cumulative = (1 + returns).cumprod()
        ax.plot(cumulative.index, cumulative.values,
                label=name, color=colors.get(name, "#333"), linewidth=2)

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Cumulative Wealth", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _drawdown_chart_base64(strategy_returns: dict[str, pd.Series]) -> str:
    """Render drawdown chart and return as base64-encoded PNG."""
    colors = {
        "PCA_SUB": "#e74c3c",
        "DOUBLE": "#3498db",
        "PCA_PLAIN": "#2ecc71",
        "MOM": "#95a5a6",
    }

    fig, ax = plt.subplots(figsize=(14, 4))
    for name, returns in strategy_returns.items():
        cumulative = (1 + returns).cumprod()
        drawdown = (cumulative / cumulative.cummax() - 1) * 100
        ax.fill_between(drawdown.index, drawdown.values, 0,
                        alpha=0.15, color=colors.get(name, "#333"))
        ax.plot(drawdown.index, drawdown.values,
                label=name, color=colors.get(name, "#333"), linewidth=1.2)

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Drawdown (%)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _highlight_best(val, col_values, higher_better=True):
    """Return CSS class if this value is the best in the column."""
    if higher_better:
        return "best" if val == max(col_values) else ""
    else:
        return "best" if val == min(col_values) else ""


def generate_html_report(
    strategy_returns: dict[str, pd.Series],
    summary_df: pd.DataFrame,
    params: dict,
    output_path: str = "report.html",
) -> None:
    """Generate a self-contained HTML report with charts and tables."""

    cumulative_img = _cumulative_chart_base64(strategy_returns)
    drawdown_img = _drawdown_chart_base64(strategy_returns)

    # Build summary table rows with best-value highlighting
    table_rows = ""
    for strategy in summary_df.index:
        row = summary_df.loc[strategy]
        cells = ""
        for col in summary_df.columns:
            val = row[col]
            col_vals = summary_df[col].tolist()
            # R/R and AR are higher-better; RISK and MDD are lower-better (less negative)
            if col in ("R/R", "AR (%)"):
                is_best = val == max(col_vals)
            else:
                is_best = val == min(col_vals)
            cls = ' class="best"' if is_best else ""
            cells += f"<td{cls}>{val}</td>"
        table_rows += f"<tr><td class='strategy-name'>{strategy}</td>{cells}</tr>\n"

    # Build monthly returns table for PCA_SUB
    pca_sub_ret = strategy_returns.get("PCA_SUB")
    yearly_table = ""
    if pca_sub_ret is not None and len(pca_sub_ret) > 0:
        monthly = (1 + pca_sub_ret).resample("ME").prod() - 1
        yearly = monthly.groupby(monthly.index.year).apply(
            lambda x: (1 + x).prod() - 1
        ) * 100
        yearly_table = "<tr>"
        for year in yearly.index:
            val = yearly.loc[year]
            color = "#27ae60" if val >= 0 else "#e74c3c"
            yearly_table += f'<td style="color:{color}; font-weight:600">{val:.1f}%</td>'
        yearly_table += "</tr>"
        yearly_header = "".join(f"<th>{y}</th>" for y in yearly.index)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Lead-Lag Strategy Report</title>
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
    padding: 2rem; max-width: 1100px; margin: 0 auto;
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
    display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 0.8rem; margin-bottom: 0.5rem;
  }}
  .param {{ text-align: center; }}
  .param-value {{ font-size: 1.4rem; font-weight: 700; color: var(--accent); }}
  .param-label {{ font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; }}
  table {{
    width: 100%; border-collapse: collapse; font-size: 0.9rem;
  }}
  th, td {{
    padding: 0.6rem 1rem; text-align: right; border-bottom: 1px solid var(--border);
  }}
  th {{ color: var(--text-muted); font-weight: 600; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.03em; }}
  td.strategy-name {{ text-align: left; font-weight: 600; color: var(--accent); }}
  td.best {{ color: var(--green); font-weight: 700; }}
  img.chart {{ width: 100%; border-radius: 6px; }}
  .footer {{
    text-align: center; color: var(--text-muted); font-size: 0.75rem;
    margin-top: 2rem; padding-top: 1rem; border-top: 1px solid var(--border);
  }}
  .formula {{
    background: #1c2128; border: 1px solid var(--border); border-radius: 6px;
    padding: 1rem 1.2rem; font-family: "SF Mono", "Fira Code", monospace;
    font-size: 0.85rem; color: #e6edf3; overflow-x: auto; margin: 0.8rem 0;
    white-space: pre;
  }}
</style>
</head>
<body>

<h1>US-Japan Sector Lead-Lag Strategy</h1>
<p class="subtitle">
  部分空間正則化付き主成分分析を用いた日米業種リードラグ投資戦略<br>
  Nakagawa et al., SIG-FIN-036, 2026 &mdash; Report generated {now}
</p>

<!-- Parameters -->
<div class="card">
  <h2 style="margin-top:0">Parameters</h2>
  <div class="params">
    <div class="param"><div class="param-value">{params.get('lambda', 0.9)}</div><div class="param-label">&lambda; (regularization)</div></div>
    <div class="param"><div class="param-value">{params.get('k', 3)}</div><div class="param-label">K (components)</div></div>
    <div class="param"><div class="param-value">{params.get('window', 60)}</div><div class="param-label">L (window)</div></div>
    <div class="param"><div class="param-value">{params.get('q', 0.3)}</div><div class="param-label">q (quantile)</div></div>
    <div class="param"><div class="param-value">{params.get('n_days', '—')}</div><div class="param-label">Trading Days</div></div>
  </div>
</div>

<!-- Key Formula -->
<div class="card">
  <h2 style="margin-top:0">Core Algorithm</h2>
  <div class="formula">C_reg(t) = (1 - &lambda;) C_t  +  &lambda; C_0          ... Regularized correlation (eq 13)
V_t = eig(C_reg(t))[:, :K]                ... Top-K eigenvectors    (eq 14)
B_t = V_JP(t) &middot; V_US(t)&rsquo;                     ... Propagation matrix     (eq 21)
signal(t) = B_t &middot; z_US(t)                    ... Predicted JP returns   (eq 19-20)</div>
</div>

<!-- Summary Table -->
<div class="card">
  <h2 style="margin-top:0">Performance Summary (Table 2)</h2>
  <table>
    <thead>
      <tr>
        <th style="text-align:left">Strategy</th>
        {"".join(f'<th>{col}</th>' for col in summary_df.columns)}
      </tr>
    </thead>
    <tbody>
      {table_rows}
    </tbody>
  </table>
</div>

<!-- Cumulative Returns Chart -->
<div class="card">
  <h2 style="margin-top:0">Cumulative Returns (Figure 2)</h2>
  <img class="chart" src="data:image/png;base64,{cumulative_img}" alt="Cumulative Returns">
</div>

<!-- Drawdown Chart -->
<div class="card">
  <h2 style="margin-top:0">Drawdown</h2>
  <img class="chart" src="data:image/png;base64,{drawdown_img}" alt="Drawdown">
</div>

<!-- Yearly Returns -->
{"" if not yearly_table else f'''
<div class="card">
  <h2 style="margin-top:0">PCA_SUB Yearly Returns</h2>
  <table>
    <thead><tr>{yearly_header}</tr></thead>
    <tbody>{yearly_table}</tbody>
  </table>
</div>
'''}

<!-- Methodology -->
<div class="card">
  <h2 style="margin-top:0">Strategies</h2>
  <table>
    <thead><tr><th style="text-align:left">Name</th><th style="text-align:left">Description</th></tr></thead>
    <tbody>
      <tr><td class="strategy-name">PCA_SUB</td><td style="text-align:left">Subspace-regularized PCA (&lambda;={params.get('lambda', 0.9)}), prior from global / country / cyclical-defensive factors</td></tr>
      <tr><td class="strategy-name">PCA_PLAIN</td><td style="text-align:left">Standard PCA without regularization (&lambda;=0)</td></tr>
      <tr><td class="strategy-name">MOM</td><td style="text-align:left">Simple momentum: rolling mean of JP close-to-close returns</td></tr>
      <tr><td class="strategy-name">DOUBLE</td><td style="text-align:left">2&times;2 double sort on MOM &times; PCA_SUB signals (median split)</td></tr>
    </tbody>
  </table>
</div>

<div class="footer">
  Reference: 部分空間正則化付き主成分分析を用いた日米業種リードラグ投資戦略 (Nakagawa et al., SIG-FIN-036, 2026)
</div>

</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"HTML report saved to: {output_path}")
