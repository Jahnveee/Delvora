import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Delivery Digital Twin", layout="wide")

# ------------------ STYLING ------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #020617);
}
h1, h2, h3 {
    color: #e2e8f0;
}
.stButton>button {
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    color: white;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ------------------ ICON HEADER ------------------
def section(title, icon):
    st.markdown(f"""
    <h3 style="display:flex; align-items:center; gap:10px;">
        <img src="https://cdn.jsdelivr.net/npm/lucide-static@latest/icons/{icon}.svg" width="20">
        {title}
    </h3>
    """, unsafe_allow_html=True)

# ------------------ DATA + MODEL ------------------
np.random.seed(42)
n = 2000

data = pd.DataFrame({
    'distance_km': np.random.randint(1, 40, n),
    'priority': np.random.choice(['low','medium','high'], n),
    'warehouse_load': np.random.choice(['low','medium','high'], n),
    'processing_time_min': np.random.randint(10,120,n),
    'dispatch_delay_min': np.random.randint(0,60,n),
    'daily_order_volume': np.random.randint(50,500,n),
    'agent_availability': np.random.choice(['low','medium','high'], n),
    'traffic_level': np.random.choice(['low','medium','high'], n),
    'weather': np.random.choice(['clear','rain','storm'], n),
    'peak_hour': np.random.choice([0,1], n)
})

# ✅ Improved realistic scoring
def calculate_delay(row):
    score = 0

    if row['distance_km'] > 25: score += 2
    elif row['distance_km'] > 15: score += 1

    if row['traffic_level'] == "high": score += 2
    elif row['traffic_level'] == "medium": score += 1

    if row['warehouse_load'] == "high": score += 2

    if row['weather'] == "storm": score += 2
    elif row['weather'] == "rain": score += 1

    if row['agent_availability'] == "low": score += 2
    elif row['agent_availability'] == "medium": score += 1

    if row['dispatch_delay_min'] > 30: score += 1
    if row['peak_hour'] == 1: score += 1

    if row['priority'] == "high": score -= 1

    return 1 if score >= 5 else 0

data['delay'] = data.apply(calculate_delay, axis=1)

data_enc = pd.get_dummies(data)
X = data_enc.drop("delay", axis=1)
y = data_enc["delay"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# ------------------ PREDICTION ------------------
def predict_delay(input_dict):
    df = pd.DataFrame([input_dict])
    df = pd.get_dummies(df)
    df = df.reindex(columns=X.columns, fill_value=0)

    prob = model.predict_proba(df)[0][1]

    # ✅ Diminishing returns (real-world behavior)
    if input_dict['agent_availability'] == 'high':
        prob *= 0.8
    if input_dict['dispatch_delay_min'] < 15:
        prob *= 0.9

    # ✅ realistic bounds
    # smooth scaling instead of hard cap
    prob = 0.05 + (prob * 0.9)

    return prob

def risk_label(p):
    return "Low" if p < 0.3 else "Medium" if p < 0.6 else "High"

# ------------------ CONTRIBUTION ENGINE ------------------
def get_feature_contributions(input_data):
    contributions = {}

    if input_data['distance_km'] > 25:
        contributions['Long Distance'] = 2
    elif input_data['distance_km'] > 15:
        contributions['Moderate Distance'] = 1

    if input_data['traffic_level'] == 'high':
        contributions['High Traffic'] = 2
    elif input_data['traffic_level'] == 'medium':
        contributions['Moderate Traffic'] = 1

    if input_data['warehouse_load'] == 'high':
        contributions['Warehouse Load'] = 2

    if input_data['weather'] == 'storm':
        contributions['Severe Weather'] = 2
    elif input_data['weather'] == 'rain':
        contributions['Rain Impact'] = 1

    if input_data['agent_availability'] == 'low':
        contributions['Low Agent Availability'] = 2
    elif input_data['agent_availability'] == 'medium':
        contributions['Limited Agents'] = 1

    if input_data['dispatch_delay_min'] > 30:
        contributions['High Dispatch Delay'] = 2

    return contributions

def classify_impact(score):
    if score >= 2:
        return "🔥 High Impact"
    elif score == 1:
        return "⚠️ Moderate Impact"
    else:
        return "Low Impact"

# ------------------ SIDEBAR ------------------
with st.sidebar:
    section("Controls", "sliders")

    distance = st.slider("Distance", 1, 40, 20)
    traffic = st.selectbox("Traffic", ["low","medium","high"])
    warehouse = st.selectbox("Warehouse", ["low","medium","high"])
    agent = st.selectbox("Agents", ["low","medium","high"])
    weather = st.selectbox("Weather", ["clear","rain","storm"])

    dispatch = st.slider("Dispatch Delay", 0, 60, 20)
    processing = st.slider("Processing", 10, 120, 60)
    priority = st.selectbox("Priority", ["low","medium","high"])
    peak = st.selectbox("Peak Hour", [0,1])
    volume = st.slider("Orders", 50, 500, 200)

    run = st.button("Run Simulation")

input_data = {
    'distance_km': distance,
    'priority': priority,
    'warehouse_load': warehouse,
    'processing_time_min': processing,
    'dispatch_delay_min': dispatch,
    'daily_order_volume': volume,
    'agent_availability': agent,
    'traffic_level': traffic,
    'weather': weather,
    'peak_hour': peak
}

# ------------------ COMPUTE ------------------
results = {}
base_prob = None

if run:
    base_prob = predict_delay(input_data)

    for name, mod in {
        "Reroute": {"traffic_level": "high"},
        "More Agents": {"agent_availability": "high"},
        "Fast Dispatch": {"dispatch_delay_min": max(0, dispatch - 20)},
        "Priority": {"priority": "high"}
    }.items():
        temp = input_data.copy()
        temp.update(mod)
        results[name] = predict_delay(temp)

# ------------------ UI ------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400..900&display=swap');

.delvora-title {
    font-family: 'Orbitron', sans-serif;
    font-weight: 700;
    letter-spacing: 1px;
}
</style>

<div style="display:flex; align-items:center; gap:12px;">
    <img src="https://cdn.jsdelivr.net/npm/lucide-static@latest/icons/package.svg" width="34">
    <h1 class="delvora-title" style="margin:0;">DELVORA</h1>
</div>

<p style="
    margin-top:6px;
    font-size:18px;
    color:#9ca3af;
    font-weight:500;
">
AI-powered Digital Twin for Delivery Optimization
</p>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Overview","Insights"])

# ------------------ OVERVIEW ------------------
with tab1:
    if base_prob is not None:

        best = min(results, key=results.get)

        st.success(
            f"System Insight: Operations are {'stable' if base_prob < 0.3 else 'at risk'}. "
            f"Best lever: {best}."
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Delay Risk", f"{round(base_prob*100,2)}%", risk_label(base_prob))
        c2.metric("Distance", distance)
        c3.metric("Orders", volume)

        st.progress(base_prob)

        section("Scenario Comparison", "bar-chart")

        cols = st.columns(len(results))

        for i, (k, v) in enumerate(results.items()):
            if k == best:
                cols[i].success(f"{k}\n{round(v*100,2)}%")
            else:
                cols[i].metric(k, f"{round(v*100,2)}%", risk_label(v))

        best_prob = results[best]
        improvement = max(0, base_prob - best_prob)

        # ✅ smarter messaging
        if improvement < 0.01:
            msg = "⚖️ Minimal impact — already optimized"
        elif improvement < 0.1:
            msg = "📉 Moderate improvement expected"
        else:
            msg = "🚀 Significant improvement opportunity"

        st.markdown(f"""
        <div style="background:#065f46;padding:22px;border-radius:12px;
                    text-align:center;color:white;font-size:18px;margin-top:10px">
        Best Action: {best} → {round(best_prob*100,2)}%<br>
        {msg}
        </div>
        """, unsafe_allow_html=True)

    else:
        st.info("Adjust controls and run simulation")

# ------------------ INSIGHTS ------------------
with tab2:
    if base_prob is not None:

        section("Why is delay happening?", "brain")

        contributions = get_feature_contributions(input_data)

        if contributions:
            sorted_factors = sorted(contributions.items(), key=lambda x: x[1], reverse=True)

            st.markdown("### 🔥 Key Risk Drivers")

            for factor, score in sorted_factors:
                impact = classify_impact(score)

                st.markdown(f"""
                <div style="
                    background:#111827;
                    padding:10px;
                    border-radius:8px;
                    margin-bottom:8px;
                    border-left:4px solid #ef4444;
                ">
                <b>{factor}</b><br>
                {impact}
                </div>
                """, unsafe_allow_html=True)

            top_factor = sorted_factors[0][0]

            st.success(f"""
Operational Insight:

The primary driver of delay is **{top_factor}**.  
Addressing this will yield the highest improvement.
""")

        for k, v in results.items():
            if abs(v - base_prob) < 0.01:
                st.warning(f"{k} has minimal impact")

        best = min(results, key=results.get)
        best_prob = results[best]
        improvement = max(0, base_prob - best_prob)

        section("Best Action Insight", "lightbulb")

        st.info(f"""
Best Strategy: {best}  
Expected Risk: {round(best_prob*100,2)}%  
Improvement: {round(improvement*100,2)}%
""")

        chart_data = pd.DataFrame({
            "Scenario": list(results.keys()),
            "Delay Risk (%)": [round(v * 100, 2) for v in results.values()]
        })

        st.bar_chart(chart_data.set_index("Scenario"))

    else:
        st.info("Run simulation to view insights")