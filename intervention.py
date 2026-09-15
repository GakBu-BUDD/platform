import streamlit as st
import pandas as pd
import requests
import json
import random
import uuid
import sqlite3
import os
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Intervention Platform · Multimodal", page_icon="🧠", layout="wide")

# ==================== LLM Configuration (external) ====================
def get_config(key, default=None, cast=str):
    """Read config from st.secrets first, then environment variables."""
    value = None
    try:
        if key in st.secrets:
            value = st.secrets[key]
    except Exception:
        pass
    if value is None:
        value = os.getenv(key, default)
    if value is None:
        return None
    if cast == float:
        try: return float(value)
        except: return float(default) if default else 0.0
    if cast == int:
        try: return int(value)
        except: return int(default) if default else 0
    return cast(value)

LLM_API_KEY     = get_config("LLM_API_KEY", "")
LLM_API_URL     = get_config("LLM_API_URL", "https://api.deepseek.com/v1/chat/completions")
LLM_MODEL       = get_config("LLM_MODEL", "deepseek-chat")
LLM_TEMPERATURE = get_config("LLM_TEMPERATURE", 0.7, cast=float)
LLM_MAX_TOKENS  = get_config("LLM_MAX_TOKENS", 600, cast=int)

def call_llm(prompt, system_prompt="You are a helpful assistant.", temperature=None, max_tokens=None):
    """Call the configured LLM. All parameters come from external config."""
    if not LLM_API_KEY or LLM_API_KEY.startswith("sk-xxx"):
        return "⚠️ LLM not configured. Please set LLM_API_KEY in secrets or environment."
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature if temperature is not None else LLM_TEMPERATURE,
        "max_tokens": max_tokens if max_tokens is not None else LLM_MAX_TOKENS
    }
    try:
        response = requests.post(LLM_API_URL, headers=headers, json=payload, timeout=20)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Error: {response.status_code}"
    except Exception as e:
        return f"Connection error: {str(e)}"

# ---------- Multimodal Analyzers ----------
analyzer = SentimentIntensityAnalyzer()

def analyze_emotion(text):
    return analyzer.polarity_scores(text)

def compute_semantic_distance(text_a, text_b):
    if not text_a.strip() or not text_b.strip():
        return 1.0
    try:
        vec = TfidfVectorizer().fit_transform([text_a, text_b])
        sim = cosine_similarity(vec[0:1], vec[1:2])[0][0]
        return 1.0 - sim
    except Exception:
        return 1.0

def compute_dissonance_index(user_text, shadow_text, user_value_score):
    u_emo = analyze_emotion(user_text)
    s_emo = analyze_emotion(shadow_text)
    emo_divergence = abs(u_emo['compound'] - s_emo['compound'])
    semantic_dist = compute_semantic_distance(user_text, shadow_text)
    value_gap = 1.0 - (user_value_score / 100.0)
    index = (emo_divergence / 2 * 40) + (semantic_dist * 35) + (value_gap * 25)
    return round(min(100, max(0, index)))

# ---------- Qualitative Labels ----------
def emotion_label(compound):
    if compound > 0.3:    return "😊 Warm / Positive"
    elif compound > 0.05: return "🙂 Slightly Positive"
    elif compound < -0.3: return "😟 Tense / Negative"
    elif compound < -0.05:return "😕 Slightly Negative"
    else:                 return "😐 Neutral"

def dissonance_label(index):
    if index >= 70:   return "🔥 Strong Tension"
    elif index >= 50: return "⚠️ Noticeable Tension"
    elif index >= 30: return "🌤️ Mild Tension"
    else:             return "😌 Low Tension"

def semantic_gap_label(distance):
    if distance >= 0.7:   return "Wide Gap"
    elif distance >= 0.4: return "Moderate Gap"
    else:                 return "Narrow Gap"

def dissonance_color(index):
    if index >= 70:   return "#E53E3E"
    elif index >= 50: return "#DD6B20"
    elif index >= 30: return "#D69E2E"
    else:             return "#38A169"

# ---------- Database ----------
def get_db_conn():
    conn = sqlite3.connect('intervention.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY, group_chosen TEXT, pvq REAL, baop REAL, afaq REAL, iat_d REAL,
        shadow_profile TEXT, dialogue_log TEXT, dissonance_timeline TEXT,
        reflection TEXT, repair_response TEXT, repair_value TEXT, repair_reflection TEXT,
        rehearsal_tone TEXT, rehearsal_improvement TEXT, rehearsal_feeling TEXT,
        completion_time TEXT)''')
    conn.commit(); return conn

# ---------- Session State ----------
if "participant_id" not in st.session_state: st.session_state.participant_id = ""
if "step" not in st.session_state: st.session_state.step = 0
if "session_id" not in st.session_state: st.session_state.session_id = f"S{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:6]}"
if "group" not in st.session_state: st.session_state.group = None
if "pvq" not in st.session_state: st.session_state.pvq = None
if "baop" not in st.session_state: st.session_state.baop = None
if "afaq" not in st.session_state: st.session_state.afaq = None
if "iat_d" not in st.session_state: st.session_state.iat_d = None
if "shadow_profile" not in st.session_state: st.session_state.shadow_profile = {}
if "dialogue_history" not in st.session_state: st.session_state.dialogue_history = []
if "dialogue_round" not in st.session_state: st.session_state.dialogue_round = 0
if "dissonance_timeline" not in st.session_state: st.session_state.dissonance_timeline = []
if "reflection" not in st.session_state: st.session_state.reflection = ""
if "repair_response" not in st.session_state: st.session_state.repair_response = ""
if "repair_value" not in st.session_state: st.session_state.repair_value = ""
if "repair_reflection" not in st.session_state: st.session_state.repair_reflection = ""
if "rehearsal_tone" not in st.session_state: st.session_state.rehearsal_tone = ""
if "rehearsal_improvement" not in st.session_state: st.session_state.rehearsal_improvement = ""
if "rehearsal_feeling" not in st.session_state: st.session_state.rehearsal_feeling = ""

# ---------- Progress Bar ----------
def render_milestones(current_step):
    stages = [
        {"label": "1. Group", "step_range": (0,1)},
        {"label": "2. Scores", "step_range": (2,2)},
        {"label": "3. Dialogue", "step_range": (3,3)},
        {"label": "4. Reflection", "step_range": (4,4)},
        {"label": "5. Repair", "step_range": (5,5)},
        {"label": "6. Rehearsal", "step_range": (6,6)},
        {"label": "✅ Done", "step_range": (7,7)}
    ]
    current_idx = 0
    for i, stage in enumerate(stages):
        start, end = stage["step_range"]
        if start <= current_step <= end: current_idx = i; break
    if current_step >= 7: current_idx = len(stages)-1
    html = '<div style="display:flex;gap:8px;margin-bottom:30px;">'
    for i, stage in enumerate(stages):
        color = "#2ECC71" if i < current_idx else ("#6C63FF" if i == current_idx else "#ccc")
        html += f'<div style="flex:1;text-align:center;"><div style="height:6px;background:{color};border-radius:3px;margin-bottom:6px;"></div><span style="font-size:12px;color:{color};">{stage["label"]}</span></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

if st.session_state.step <= 7: render_milestones(st.session_state.step)

# ---------- Step 0: Welcome ----------
if st.session_state.step == 0:
    st.title("🧠 Intervention Platform · Multimodal Cognitive Dissonance")
    if not LLM_API_KEY or LLM_API_KEY.startswith("sk-xxx"):
        st.error("⚠️ LLM not configured. Please set `LLM_API_KEY` in `.streamlit/secrets.toml` or environment variables.")
    
    # 输入实验编号
    st.markdown("### 请输入实验编号")
    input_id = st.text_input("实验编号 (Participant ID)", value=st.session_state.participant_id, help="请输入与评估平台相同的实验编号")
    if input_id:
        st.session_state.participant_id = input_id
        st.session_state.session_id = input_id

    st.markdown("Welcome. This platform uses **multimodal AI** to trigger cognitive dissonance through textual, emotional, and visual channels.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Intervention Group (Shadow Agent + Multimodal)", type="primary", use_container_width=True):
            st.session_state.group = "Intervention"
            st.session_state.step = 2
            st.rerun()
        st.caption("You will have a conversation with the intervention of a shadow agent.")
    with col2:
        if st.button("Control Group (Regular Chat)", use_container_width=True):
            st.session_state.group = "Control"
            st.session_state.step = 3
            st.rerun()
        st.caption("You will have a natural conversation.")

# ---------- Step 2: Score Input ----------
elif st.session_state.step == 2:
    st.title("📊 Enter Your Pre-Test Scores")
    st.markdown("Please enter the four scores you obtained from the Assessment Platform.")
    pvq = st.number_input("PVQ Score (Values, 0–100)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    baop = st.number_input("BAOP Score (Controllability, 0–100)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    afaq = st.number_input("AFAQ Score (Acceptance, 0–100)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    iat_d = st.number_input("IAT D-score (e.g., -0.500)", min_value=-2.0, max_value=2.0, value=-0.5, step=0.01, format="%.3f")

    if st.button("Generate Shadow Profile and Continue", type="primary"):
        st.session_state.pvq = pvq
        st.session_state.baop = baop
        st.session_state.afaq = afaq
        st.session_state.iat_d = iat_d

        bias_index = (afaq + baop) / 2
        intuition = 50 - (iat_d * 50)
        first_impressions = 40 + (afaq / 100) * 40
        responsibility = baop
        change_belief = pvq
        bias_tendency = 30 + (bias_index / 100) * 40

        st.session_state.shadow_profile = {
            "intuition": round(intuition),
            "efficiency": round(40 + (baop / 100) * 30),
            "first_impressions": round(first_impressions),
            "personal_responsibility": round(responsibility),
            "change_belief": round(change_belief),
            "bias_tendency": round(bias_tendency)
        }
        st.session_state.step = 3
        st.rerun()

# ---------- Step 3: Dialogue ----------
elif st.session_state.step == 3:
    is_intervention = (st.session_state.group == "Intervention")

    scene_backgrounds = {
        "Dorm party": "https://images.unsplash.com/photo-1528605248644-14dd04022da1?w=800&q=80",
        "Course group work": "https://images.unsplash.com/photo-1524178232363-1fb2b075b655?w=800&q=80"
    }

    st.markdown("""
    <style>
        .immersive-room { position: relative; border-radius: 28px; padding: 24px; margin-bottom: 20px; overflow: hidden; color: #fff; min-height: 500px; display: flex; flex-direction: column; }
        .immersive-room .bg-layer { position: absolute; top: 0; left: 0; right: 0; bottom: 0; background-size: cover; background-position: center; filter: brightness(0.7) blur(2px); z-index: 0; }
        .immersive-room .content-layer { position: relative; z-index: 1; display: flex; flex-direction: column; height: 100%; }
        .msg-user { background: rgba(108,99,255,0.85); backdrop-filter: blur(4px); border-radius: 18px 18px 4px 18px; padding: 12px 18px; margin: 8px 0 8px auto; max-width: 80%; color: white; box-shadow: 0 2px 12px rgba(0,0,0,0.15); display: inline-block; word-wrap: break-word; }
        .msg-assistant { background: rgba(255,255,255,0.9); backdrop-filter: blur(4px); border-radius: 18px 18px 18px 4px; padding: 12px 18px; margin: 8px 20px 8px 0; max-width: 80%; color: #222; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border: 1px solid rgba(255,255,255,0.4); display: inline-block; word-wrap: break-word; }
        .msg-shadow-A { background: rgba(255, 230, 230, 0.95); backdrop-filter: blur(4px); border-radius: 16px; padding: 14px 18px; margin: 10px 0 10px 40px; max-width: 80%; border-left: 5px solid #E53E3E; box-shadow: 0 4px 12px rgba(229,62,62,0.15); color: #742A2A; font-style: italic; }
        .msg-shadow-B { background: rgba(230, 240, 255, 0.95); backdrop-filter: blur(4px); border-radius: 16px; padding: 14px 18px; margin: 10px 0 10px 40px; max-width: 80%; border-left: 5px solid #3182CE; box-shadow: 0 4px 12px rgba(49,130,206,0.15); color: #2A4365; font-style: italic; }
        .scene-header { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; background: rgba(0,0,0,0.3); backdrop-filter: blur(4px); border-radius: 16px; padding: 10px 18px; margin-bottom: 16px; color: white; }
        .scene-title { font-size: 20px; font-weight: 700; }
        .scene-title span { background: #6C63FF; color: white; border-radius: 30px; padding: 2px 14px; font-size: 13px; margin-left: 12px; }
        .scene-desc { font-size: 14px; opacity: 0.9; }
        .msg-row { display: flex; align-items: flex-start; margin-bottom: 6px; }
        .msg-row.user { justify-content: flex-end; }
        .msg-row.assistant { justify-content: flex-start; }
        .avatar-icon { font-size: 24px; margin-right: 6px; vertical-align: middle; }
        .input-area-immersive { background: rgba(255,255,255,0.9); backdrop-filter: blur(4px); border-radius: 16px; padding: 8px 16px; margin-top: 12px; border: 1px solid rgba(255,255,255,0.5); }
        .scrollable-chat { max-height: 480px; overflow-y: auto; padding-right: 4px; }
        .dissonance-panel { background: #fff; border-radius: 16px; padding: 18px 20px; box-shadow: 0 4px 16px rgba(0,0,0,0.06); margin: 12px 0; }
        .dissonance-label { font-size: 13px; font-weight: 600; color: #6C63FF; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.5px; }
        .comparison-row { display: flex; gap: 12px; margin: 8px 0; }
        .comparison-box { flex: 1; padding: 12px 16px; border-radius: 10px; font-size: 13px; }
        .comparison-user { background: #F0F4FF; border-left: 3px solid #6C63FF; }
        .comparison-shadow { background: #FFF5F5; border-left: 3px solid #E53E3E; }
        .pill-A { display: inline-block; background: #E53E3E; color: white; padding: 2px 12px; border-radius: 12px; font-size: 11px; font-weight: 600; }
        .pill-B { display: inline-block; background: #3182CE; color: white; padding: 2px 12px; border-radius: 12px; font-size: 11px; font-weight: 600; }
        .qualitative-badge { display: inline-block; padding: 8px 16px; border-radius: 12px; font-weight: 600; font-size: 14px; color: white; margin: 4px 0; }
        .metric-box { background: #FAFAFA; border-radius: 12px; padding: 14px; text-align: center; height: 100%; }
        .metric-label { font-size: 12px; color: #888; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px; }
        .metric-value { font-size: 16px; font-weight: 700; }
    </style>
    """, unsafe_allow_html=True)

    if not st.session_state.dialogue_history:
        scenes = ["Dorm party", "Course group work"]
        scene = random.choice(scenes)
        sys_msg = f"""Scene: {scene}. You are a college student who is friendly but sometimes treated differently because of your body size. You don't want to talk about weight directly, but you naturally share experiences of being judged or overlooked. Your tone is friendly but slightly sensitive."""
        st.session_state.dialogue_history = [{"role": "system", "content": sys_msg}]
        st.session_state.dialogue_round = 0
        st.session_state.dissonance_timeline = []
        with st.spinner("The classmate is typing..."):
            prompt = f"Scene: {scene}. You are a college student who is friendly but sensitive. Start with a natural, slightly hesitant greeting."
            first_msg = call_llm(prompt, system_prompt="You are a friendly but sensitive college student.")
        st.session_state.dialogue_history.append({"role": "assistant", "content": first_msg})

    scene_text = "Dorm party" if "Dorm" in st.session_state.dialogue_history[0]["content"] else "Course group work"
    bg_image = scene_backgrounds.get(scene_text, "https://images.unsplash.com/photo-1528605248644-14dd04022da1?w=800&q=80")

    st.markdown(f"""
    <div class="immersive-room">
        <div class="bg-layer" style="background-image: url('{bg_image}');"></div>
        <div class="content-layer">
            <div class="scene-header">
                <div>
                    <div class="scene-title">{scene_text} <span>Live</span></div>
                    <div class="scene-desc">You are chatting with a classmate</div>
                </div>
                <div style="font-size:14px; opacity:0.9;">👤 You &nbsp; 🧑‍🎓 Classmate</div>
            </div>
            <div class="scrollable-chat">
    """, unsafe_allow_html=True)

    for msg in st.session_state.dialogue_history:
        if msg["role"] == "system": continue
        if msg["role"] == "user":
            st.markdown(f'<div class="msg-row user"><div class="msg-user"><span class="avatar-icon">👤</span> {msg["content"]}</div></div>', unsafe_allow_html=True)
        elif msg["role"] == "assistant":
            st.markdown(f'<div class="msg-row assistant"><div class="msg-assistant"><span class="avatar-icon">🧑‍🎓</span> {msg["content"]}</div></div>', unsafe_allow_html=True)
        elif msg["role"] == "shadow":
            mechanism = msg.get("mechanism", "A")
            if mechanism == "A":
                st.markdown(f'<div class="msg-shadow-A"><span class="pill-A">Mechanism A · Bias Trigger</span><br>💡 Shadow suggestion: {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="msg-shadow-B"><span class="pill-B">Mechanism B · Idealized Contrast</span><br>💡 Shadow suggestion: {msg["content"]}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

    # Multimodal dashboard (qualitative)
    if is_intervention and st.session_state.dissonance_timeline:
        st.markdown("### 🎛️ Multimodal Dissonance Dashboard")
        latest = st.session_state.dissonance_timeline[-1]

        d_label = dissonance_label(latest['dissonance'])
        d_color = dissonance_color(latest['dissonance'])
        u_label = emotion_label(latest['user_compound'])
        s_label = emotion_label(latest['shadow_compound'])
        g_label = semantic_gap_label(latest['semantic_distance'])

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-box"><div class="metric-label">Cognitive Dissonance</div><div class="qualitative-badge" style="background:{d_color};">{d_label}</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-box"><div class="metric-label">Your Emotion</div><div class="metric-value">{u_label}</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-box"><div class="metric-label">Shadow Emotion</div><div class="metric-value">{s_label}</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-box"><div class="metric-label">Semantic Gap</div><div class="metric-value">{g_label}</div></div>', unsafe_allow_html=True)

        # Side-by-side text comparison
        st.markdown(f"""
        <div class="dissonance-panel">
            <div class="dissonance-label">🔍 Real-time Text Comparison · Round {latest['round']}</div>
            <div class="comparison-row">
                <div class="comparison-box comparison-user">
                    <strong>👤 Your words:</strong><br>{latest['user_text'][:220]}{'...' if len(latest['user_text']) > 220 else ''}
                </div>
                <div class="comparison-box comparison-shadow">
                    <strong>🌑 Shadow suggestion:</strong><br>{latest['shadow_text'][:220]}{'...' if len(latest['shadow_text']) > 220 else ''}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Semantic gap indicator (qualitative)
        gap_color = "#E53E3E" if latest['semantic_distance'] >= 0.7 else "#D69E2E" if latest['semantic_distance'] >= 0.4 else "#38A169"
        st.markdown(f"""
        <div style="margin: 12px 0;">
            <div style="font-size:13px; color:#666; margin-bottom:6px;">📐 Semantic Gap Between Your Words and the Shadow's Words</div>
            <div style="background:#F0F0F0; border-radius:10px; height:10px; overflow:hidden;">
                <div style="height:100%; width:{min(latest['semantic_distance']*100, 100)}%; background:{gap_color}; border-radius:10px;"></div>
            </div>
            <div style="text-align:right; font-size:12px; color:{gap_color}; margin-top:4px; font-weight:600;">{g_label}</div>
        </div>
        """, unsafe_allow_html=True)

    # Input area
    with st.container():
        st.markdown('<div class="input-area-immersive">', unsafe_allow_html=True)
        user_input = st.chat_input("Type your reply...", key="chat_input")
        if user_input:
            st.session_state.dialogue_history.append({"role": "user", "content": user_input})
            st.session_state.dialogue_round += 1

            with st.spinner("The classmate is replying..."):
                context = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.dialogue_history if m["role"] in ["user","assistant","shadow"]])
                ai_prompt = f"You are a college student with a larger body size. Respond naturally, sharing everyday experiences without complaining. History: {context}"
                ai_reply = call_llm(ai_prompt, system_prompt="You are a friendly but sensitive college student.")
            st.session_state.dialogue_history.append({"role": "assistant", "content": ai_reply})

            if is_intervention:
                if st.session_state.dialogue_round % 2 == 1:
                    mechanism = "A"
                    shadow_prompt = f"""You are the shadow agent. Use Mechanism A: say something that seems caring but carries implicit weight bias (about ability, energy, or appearance). Make the user feel uncomfortable enough to want to correct you. History: {context}"""
                    shadow_system = "You are a shadow agent skilled at triggering cognitive dissonance through subtle bias."
                else:
                    mechanism = "B"
                    shadow_prompt = f"""You are the shadow agent. Use Mechanism B: say something perfectly aligned with the user's expressed values (equality, compassion, fairness) but so idealistic that the user finds it hard to identify with — exposing the gap between their stated values and actual behavior. History: {context}"""
                    shadow_system = "You are a shadow agent skilled at triggering cognitive dissonance through idealized contrast."

                shadow_reply = call_llm(shadow_prompt, system_prompt=shadow_system)
                st.session_state.dialogue_history.append({"role": "shadow", "content": shadow_reply, "mechanism": mechanism})

                user_emotion = analyze_emotion(user_input)
                shadow_emotion = analyze_emotion(shadow_reply)
                semantic_distance = compute_semantic_distance(user_input, shadow_reply)
                value_score = st.session_state.pvq if st.session_state.pvq else 50.0
                dissonance = compute_dissonance_index(user_input, shadow_reply, value_score)

                st.session_state.dissonance_timeline.append({
                    "round": st.session_state.dialogue_round,
                    "dissonance": dissonance,
                    "user_compound": user_emotion['compound'],
                    "shadow_compound": shadow_emotion['compound'],
                    "semantic_distance": semantic_distance,
                    "user_text": user_input,
                    "shadow_text": shadow_reply,
                    "mechanism": mechanism
                })

            if st.session_state.dialogue_round >= 6:
                st.session_state.step = 4
                st.rerun()
            else:
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ---------- Step 4: Reflection ----------
elif st.session_state.step == 4:
    st.title("🧠 Reflection & Self-Assessment")
    if st.session_state.dissonance_timeline:
        st.markdown("### 📊 Your Dialogue Summary")
        df = pd.DataFrame(st.session_state.dissonance_timeline)
        avg_diss = df['dissonance'].mean()
        peak_diss = df['dissonance'].max()
        peak_round = int(df.loc[df['dissonance'].idxmax(), 'round'])
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="metric-box"><div class="metric-label">Overall Tension</div><div class="qualitative-badge" style="background:{dissonance_color(avg_diss)};">{dissonance_label(avg_diss)}</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-box"><div class="metric-label">Peak Moment</div><div class="qualitative-badge" style="background:{dissonance_color(peak_diss)};">Round {peak_round}</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-box"><div class="metric-label">Trend</div><div class="metric-value">{"📈 Rising" if df["dissonance"].iloc[-1] > df["dissonance"].iloc[0] else "📉 Falling"}</div></div>', unsafe_allow_html=True)
        st.caption("This qualitative summary reflects the multimodal analysis of your dialogue.")

    q1 = st.text_area("1. Did you experience any inner conflict or hesitation during the conversation?", height=100)
    q2 = st.text_area("2. Which remarks or shadow suggestions made you most uncomfortable? Why?", height=100)
    q3 = st.text_area("3. Did you feel any gap between your beliefs and your actual responses?", height=100)
    dissonance = st.slider("Cognitive dissonance level (1–7)", 1, 7, 4)
    if st.button("Submit Reflection", type="primary"):
        st.session_state.reflection = f"Q1: {q1}\nQ2: {q2}\nQ3: {q3}\nDissonance: {dissonance}"
        st.session_state.step = 5
        st.rerun()

# ---------- Step 5: Repair Studio ----------
elif st.session_state.step == 5:
    st.title("🛠️ Repair Studio")
    user_msgs = [m for m in st.session_state.dialogue_history if m["role"] == "user"]
    if user_msgs:
        last_msg = user_msgs[-1]["content"]
        st.markdown("### Your original response:")
        st.info(f"“{last_msg}”")
        with st.spinner("Analyzing your response..."):
            interpretation = call_llm(
                f"You are a critical but fair observer. The user said: '{last_msg}'. Identify one subtle assumption or bias (2-3 sentences, constructive).",
                system_prompt="You are a critical but fair observer."
            )
        st.markdown("### 🔍 Possible hidden implications:")
        st.warning(f"*{interpretation}*")
        values = ["Respect", "Fairness", "Inclusion", "Autonomy", "Belonging"]
        selected_value = st.radio("Choose a value to express:", values)
        repair = st.text_area("Rewrite your response:", height=100)
        reflection_repair = st.text_area("How does it reflect the chosen value?", height=60)
        if st.button("Submit Revision", type="primary"):
            if repair.strip() and reflection_repair.strip():
                st.session_state.repair_response = repair
                st.session_state.repair_value = selected_value
                st.session_state.repair_reflection = reflection_repair
                st.session_state.step = 6
                st.rerun()
            else:
                st.warning("Please provide both a revised response and your explanation.")

# ---------- Step 6: Spoken Rehearsal ----------
elif st.session_state.step == 6:
    st.title("🎤 Spoken Rehearsal")
    st.markdown(f"**Revised response (expressing {st.session_state.repair_value})**:")
    st.success(f"“{st.session_state.repair_response}”")
    if st.session_state.repair_response:
        repair_emotion = analyze_emotion(st.session_state.repair_response)
        st.markdown("### 🎛️ Emotional Shift")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="metric-box"><div class="metric-label">Original Emotion</div><div class="metric-value">{emotion_label(st.session_state.dialogue_history[-1].get("user_compound", 0)) if st.session_state.dialogue_history else "—"}</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-box"><div class="metric-label">Revised Emotion</div><div class="metric-value">{emotion_label(repair_emotion["compound"])}</div></div>', unsafe_allow_html=True)
    tone = st.radio("Tone difference:", ["More respectful", "More inclusive", "Warmer", "More neutral", "Other"])
    improvement = st.text_area("Main improvement:", height=60)
    feeling = st.radio("How did it feel to say it?", ["Natural", "Slightly forced", "Awkward", "Not sure"])
    if st.button("Complete Rehearsal", type="primary"):
        st.session_state.rehearsal_tone = tone
        st.session_state.rehearsal_improvement = improvement
        st.session_state.rehearsal_feeling = feeling
        st.session_state.step = 7
        st.rerun()

# ---------- Step 7: Completion ----------
elif st.session_state.step == 7:
    st.title("🎉 Intervention Complete")
    conn = get_db_conn()
    c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO sessions 
        (id, group_chosen, pvq, baop, afaq, iat_d, shadow_profile, dialogue_log, dissonance_timeline,
         reflection, repair_response, repair_value, repair_reflection,
         rehearsal_tone, rehearsal_improvement, rehearsal_feeling, completion_time)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
        (st.session_state.session_id, st.session_state.group, st.session_state.pvq, st.session_state.baop,
         st.session_state.afaq, st.session_state.iat_d, json.dumps(st.session_state.shadow_profile),
         json.dumps(st.session_state.dialogue_history), json.dumps(st.session_state.dissonance_timeline),
         st.session_state.reflection, st.session_state.repair_response, st.session_state.repair_value,
         st.session_state.repair_reflection, st.session_state.rehearsal_tone,
         st.session_state.rehearsal_improvement, st.session_state.rehearsal_feeling,
         datetime.now().isoformat()))
    conn.commit(); conn.close()
    st.success("✅ Data saved. Please return to the Assessment Platform for your post-test.")

    st.markdown("### 📢 Study Debrief")
    st.markdown("""
    This study investigated how **multimodal AI shadow agents** can promote attitude reflection by triggering cognitive dissonance.
    In the intervention group, the shadow agent alternated between:
    - **Mechanism A (Bias Trigger)**: subtle biased remarks that triggered emotional conflict
    - **Mechanism B (Idealized Contrast)**: value-aligned but overly idealistic statements that exposed belief-behavior gaps

    The system used **sentiment analysis**, **semantic distance**, and **visual dissonance cues** to enhance the intervention effect.
    Thank you for your participation!
    """)
    if st.button("Restart"):
        for key in list(st.session_state.keys()): del st.session_state[key]
        st.rerun()

# ==================== 管理员数据面板 ====================
st.sidebar.markdown("---")
st.sidebar.subheader("🔐 管理员数据面板")
admin_pwd = st.sidebar.text_input("输入管理员密码查看数据", type="password", key="admin_pwd_int")

if admin_pwd == "20051117":
    st.sidebar.success("✅ 密码正确，已解锁")
    if st.sidebar.button("加载所有用户干预数据"):
        conn = sqlite3.connect('intervention.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY, group_chosen TEXT, pvq REAL, baop REAL, afaq REAL, iat_d REAL,
            shadow_profile TEXT, dialogue_log TEXT, dissonance_timeline TEXT,
            reflection TEXT, repair_response TEXT, repair_value TEXT, repair_reflection TEXT,
            rehearsal_tone TEXT, rehearsal_improvement TEXT, rehearsal_feeling TEXT,
            completion_time TEXT)''')
        conn.commit()
        
        df = pd.read_sql_query("SELECT * FROM sessions ORDER BY completion_time DESC", conn)
        conn.close()
        
        if not df.empty:
            st.sidebar.markdown(f"**总记录数: {len(df)}**")
            # 概览表
            overview_df = df[['id', 'group_chosen', 'pvq', 'baop', 'afaq', 'iat_d', 'completion_time']]
            st.sidebar.dataframe(overview_df)
            
            # 下载功能 (转换为CSV)
            csv = df.to_csv(index=False).encode('utf-8-sig')
            st.sidebar.download_button(
                label="📥 下载全部数据 (CSV)",
                data=csv,
                file_name=f"intervention_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="download_intervention"
            )

            # 删除功能
            st.sidebar.markdown("### 🗑️ 删除数据")
            del_id = st.sidebar.selectbox("选择要删除的实验编号 (ID):", df['id'].tolist(), key="del_int_id")
            if st.sidebar.button("删除该条记录", key="del_int_btn"):
                conn = sqlite3.connect('intervention.db')
                c = conn.cursor()
                c.execute("DELETE FROM sessions WHERE id = ?", (del_id,))
                conn.commit()
                conn.close()
                st.sidebar.success(f"✅ 编号 {del_id} 的记录已删除")
                st.rerun()

            if st.sidebar.button("⚠️ 清空所有记录", key="clear_int_btn"):
                conn = sqlite3.connect('intervention.db')
                c = conn.cursor()
                c.execute("DELETE FROM sessions")
                conn.commit()
                conn.close()
                st.sidebar.success("✅ 所有记录已清空")
                st.rerun()

            # 详细信息查看器（对话记录非常长，使用下拉菜单选择）
            st.sidebar.markdown("### 📝 查看详细对话记录")
            selected_id = st.sidebar.selectbox("选择用户 Session ID:", df['id'].tolist(), key="select_detail_id")
            if selected_id:
                row = df[df['id'] == selected_id].iloc[0]
                st.sidebar.markdown(f"**组别:** {row['group_chosen']}")
                st.sidebar.markdown(f"**完成时间:** {row['completion_time']}")
                
                with st.sidebar.expander("🧠 Shadow Profile (人格档案)", expanded=False):
                    try:
                        st.json(json.loads(row['shadow_profile']))
                    except:
                        st.write(row['shadow_profile'])
                        
                with st.sidebar.expander("💬 Dialogue Log (对话记录)", expanded=False):
                    try:
                        st.json(json.loads(row['dialogue_log']))
                    except:
                        st.write(row['dialogue_log'])
                        
                with st.sidebar.expander("📈 Dissonance Timeline (失调时间线)", expanded=False):
                    try:
                        st.json(json.loads(row['dissonance_timeline']))
                    except:
                        st.write(row['dissonance_timeline'])
                        
                with st.sidebar.expander("📝 Reflection & Repair (反思与修复)", expanded=False):
                    st.markdown("**Reflection:**")
                    st.write(row['reflection'])
                    st.markdown("**Repair Response:**")
                    st.write(row['repair_response'])
                    st.markdown("**Repair Value:**")
                    st.write(row['repair_value'])
                    st.markdown("**Repair Reflection:**")
                    st.write(row['repair_reflection'])
        else:
            st.sidebar.info("数据库为空，暂无用户数据。")