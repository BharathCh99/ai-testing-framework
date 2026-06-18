import ollama
import json
import math
import time
from datetime import datetime, timedelta
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models.base_model import DeepEvalBaseLLM

class OllamaModel(DeepEvalBaseLLM):
    def load_model(self): return "llama3"
    def generate(self, prompt: str) -> str:
        r = ollama.chat(model="llama3",
            messages=[{"role":"user","content":prompt}])
        return r['message']['content']
    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)
    def get_model_name(self): return "llama3"

# ── Statistical Functions ────────────────────────────────

def calculate_mean(values: list) -> float:
    return round(sum(values) / len(values), 4)

def calculate_std_dev(values: list) -> float:
    mean = calculate_mean(values)
    variance = sum((x - mean) ** 2 
                   for x in values) / len(values)
    return round(math.sqrt(variance), 4)

def calculate_zscore(value: float, 
                     mean: float, 
                     std: float) -> float:
    if std == 0:
        return 0
    return round((value - mean) / std, 4)

def detect_anomaly(zscore: float) -> dict:
    """Classify drift severity using Z-score"""
    abs_z = abs(zscore)
    if abs_z < 1.5:
        return {"level": "NORMAL", "color": "GREEN"}
    elif abs_z < 2.0:
        return {"level": "WARNING", "color": "YELLOW"}
    elif abs_z < 2.5:
        return {"level": "CRITICAL", "color": "ORANGE"}
    else:
        return {"level": "SEVERE", "color": "RED"}

# ── Simulate Historical Data ─────────────────────────────

def simulate_historical_scores() -> list:
    """Simulate 14 days of quality scores"""
    # Days 1-10: Good baseline (0.85-1.0)
    # Days 11-12: Slight degradation
    # Days 13-14: Significant drift
    simulated = [
	    {"day": 1,  "score": 1.0,  "date": "2026-06-01"},
        {"day": 2,  "score": 0.95, "date": "2026-06-02"},
        {"day": 3,  "score": 1.0,  "date": "2026-06-03"},
        {"day": 4,  "score": 0.90, "date": "2026-06-04"},
        {"day": 5,  "score": 0.95, "date": "2026-06-05"},
        {"day": 6,  "score": 1.0,  "date": "2026-06-06"},
        {"day": 7,  "score": 0.90, "date": "2026-06-07"},
        {"day": 8,  "score": 0.95, "date": "2026-06-08"},
        {"day": 9,  "score": 1.0,  "date": "2026-06-09"},
        {"day": 10, "score": 0.90, "date": "2026-06-10"},
        {"day": 11, "score": 0.75, "date": "2026-06-11"},
        {"day": 12, "score": 0.65, "date": "2026-06-12"},
        {"day": 13, "score": 0.45, "date": "2026-06-13"},
        {"day": 14, "score": 0.30, "date": "2026-06-14"},
    ]
    return simulated

# ── Get Today's Real Score ───────────────────────────────

def get_todays_score() -> float:
    """Get actual quality score from llama3 today"""
    print("Getting today's live score from llama3...")
    ollama_model = OllamaModel()
    metric = AnswerRelevancyMetric(
        threshold=0.5, model=ollama_model)

    test_case = LLMTestCase(
        input="What is RHB fixed deposit interest rate?",
        actual_output=ollama.chat(
            model="llama3",
            messages=[{"role": "user",
                       "content": "What is RHB fixed deposit interest rate?"}]
        )['message']['content']
    )
    metric.measure(test_case)
    return round(metric.score, 4)

# ── Advanced Drift Analysis ──────────────────────────────

def analyze_drift(historical: list, 
                  today_score: float) -> dict:
    """Full statistical drift analysis"""

    # Use first 10 days as baseline
    baseline_scores = [d['score'] 
                       for d in historical[:10]]
    recent_scores = [d['score'] 
                     for d in historical[10:]]

    baseline_mean = calculate_mean(baseline_scores)
    baseline_std = calculate_std_dev(baseline_scores)
    recent_mean = calculate_mean(recent_scores)

    # Z-score for today's score against baseline
    today_zscore = calculate_zscore(
        today_score, baseline_mean, baseline_std)
    anomaly = detect_anomaly(today_zscore)

    # Trend analysis
    trend = "IMPROVING" if today_score > recent_mean \
        else "DEGRADING"

    # Drift percentage
    drift_pct = round(
        ((today_score - baseline_mean) / baseline_mean) * 100, 1)

    return {
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "recent_mean": recent_mean,
        "today_score": today_score,
        "today_zscore": today_zscore,
        "anomaly": anomaly,
        "trend": trend,
        "drift_percentage": drift_pct
    }

# ── Generate HTML Dashboard ──────────────────────────────

def generate_drift_dashboard(historical: list,
                              analysis: dict):
    """Generate advanced drift monitoring dashboard"""
    today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build chart data
    labels = [f"Day {d['day']}" for d in historical]
    scores = [d['score'] for d in historical]
    labels.append("Today")
    scores.append(analysis['today_score'])

    labels_js = str(labels).replace("'", '"')
    scores_js = str(scores)

    anomaly = analysis['anomaly']
    alert_color = {
        "GREEN": "#28a745",
        "YELLOW": "#ffc107",
        "ORANGE": "#fd7e14",
        "RED": "#dc3545"
    }.get(anomaly['color'], "#dc3545")

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Advanced Drift Detection Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js">
    </script>
    <style>
        body {{ font-family: Arial, sans-serif;
               margin: 40px; background: #0d1117;
               color: #e6edf3; }}
        h1 {{ color: #58a6ff; }}
        .grid {{ display: grid;
                 grid-template-columns: repeat(4, 1fr);
                 gap: 15px; margin: 20px 0; }}
        .card {{ background: #161b22; padding: 20px;
                 border-radius: 8px;
                 border: 1px solid #30363d;
                 text-align: center; }}
        .card h2 {{ font-size: 2em; margin: 0; }}
        .card p {{ color: #8b949e; margin: 5px 0 0; }}
        .alert {{ padding: 20px; border-radius: 8px;
                  margin: 20px 0; text-align: center;
                  font-size: 1.5em; font-weight: bold;
                  background: {alert_color}22;
                  border: 2px solid {alert_color};
                  color: {alert_color}; }}
        .chart-container {{ background: #161b22;
                            padding: 20px;
                            border-radius: 8px;
                            margin: 20px 0; }}
        .stats {{ display: grid;
                  grid-template-columns: 1fr 1fr;
                  gap: 15px; margin: 20px 0; }}
        .stat-card {{ background: #161b22; padding: 15px;
                      border-radius: 8px;
                      border: 1px solid #30363d; }}
        .stat-card h3 {{ color: #58a6ff; margin-top: 0; }}
    </style>
</head>
<body>
    <h1>📊 Advanced AI Drift Detection Dashboard</h1>
    <p>Model: <strong>llama3 (Ollama)</strong> |
       Generated: <strong>{today}</strong></p>

    <div class="alert">
        {anomaly['level']} — 
        Drift: {analysis['drift_percentage']}% | 
        Trend: {analysis['trend']}
    </div>

    <div class="grid">
        <div class="card">
            <h2 style="color:#58a6ff">
                {analysis['baseline_mean']}</h2>
            <p>Baseline Mean (10 days)</p>
        </div>
        <div class="card">
            <h2 style="color:{alert_color}">
                {analysis['today_score']}</h2>
            <p>Today's Score</p>
        </div>
        <div class="card">
            <h2 style="color:#ffc107">
                {analysis['today_zscore']}</h2>
            <p>Z-Score</p>
        </div>
        <div class="card">
            <h2 style="color:{alert_color}">
                {analysis['drift_percentage']}%</h2>
            <p>Drift from Baseline</p>
        </div>
    </div>

    <div class="chart-container">
        <canvas id="driftChart"></canvas>
    </div>

    <div class="stats">
        <div class="stat-card">
            <h3>Statistical Summary</h3>
            <p>Baseline Mean: 
               <strong>{analysis['baseline_mean']}</strong></p>
            <p>Baseline Std Dev: 
               <strong>{analysis['baseline_std']}</strong></p>
            <p>Recent Mean: 
               <strong>{analysis['recent_mean']}</strong></p>
            <p>Today Z-Score: 
               <strong>{analysis['today_zscore']}</strong></p>
        </div>
        <div class="stat-card">
            <h3>Alert Thresholds</h3>
            <p>🟢 Normal: Z-Score &lt; 1.5</p>
            <p>🟡 Warning: Z-Score 1.5 - 2.0</p>
            <p>🟠 Critical: Z-Score 2.0 - 2.5</p>
            <p>🔴 Severe: Z-Score &gt; 2.5</p>
        </div>
    </div>

    <script>
    const ctx = document.getElementById(
        'driftChart').getContext('2d');
    const labels = {labels_js};
    const scores = {scores_js};
    const baseline = {analysis['baseline_mean']};

    new Chart(ctx, {{
        type: 'line',
        data: {{
            labels: labels,
            datasets: [
            {{
                label: 'Quality Score',
                data: scores,
                borderColor: '#58a6ff',
                backgroundColor: '#58a6ff22',
                tension: 0.3,
                fill: true
            }},
            {{
                label: 'Baseline Mean',
                data: new Array(labels.length).fill(baseline),
                borderColor: '#28a745',
                borderDash: [5, 5],
                pointRadius: 0
            }}
            ]
        }},
        options: {{
            responsive: true,
            plugins: {{
                title: {{
                    display: true,
                    text: 'AI Quality Score Trend (14 Days)',
                    color: '#e6edf3',
                    font: {{ size: 16 }}
                }},
                legend: {{
                    labels: {{ color: '#e6edf3' }}
                }}
            }},
            scales: {{
                y: {{
                    min: 0, max: 1,
                    grid: {{ color: '#30363d' }},
                    ticks: {{ color: '#e6edf3' }}
                }},
                x: {{
                    grid: {{ color: '#30363d' }},
                    ticks: {{ color: '#e6edf3' }}
                }}
            }}
        }}
    }});
    </script>
</body>
</html>"""

    with open("advanced_drift_dashboard.html",
              "w", encoding="utf-8") as f:
        f.write(html)
    print("✅ Dashboard saved!")

# ── Main ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  ADVANCED DRIFT DETECTOR")
    print("=" * 60)

    # Get historical + today's data
    historical = simulate_historical_scores()
    today_score = get_todays_score()
    print(f"Today's live score: {today_score}")

    # Analyze drift
    analysis = analyze_drift(historical, today_score)

    # Print summary
    print(f"\nBaseline Mean: {analysis['baseline_mean']}")
    print(f"Today Score:   {analysis['today_score']}")
    print(f"Z-Score:       {analysis['today_zscore']}")
    print(f"Status:        {analysis['anomaly']['level']}")
    print(f"Drift:         {analysis['drift_percentage']}%")
    print(f"Trend:         {analysis['trend']}")

    # Generate dashboard
    generate_drift_dashboard(historical, analysis)
    print("\nOpen: advanced_drift_dashboard.html")
    print("=" * 60)