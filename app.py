# ================================================================
# 🧠 Master AI — Multi-Model Consensus + Debate + Image + Voice
# ================================================================
# पूरा कोड एक ही file में — Chat, Debate, Image Gen, Voice, History
# ================================================================

import os
import uuid
import base64
import sqlite3
import asyncio
import tempfile
from datetime import datetime

import streamlit as st
from openai import AsyncOpenAI, OpenAI
from dotenv import load_dotenv

# ================================================================
# 1. CONFIG — Settings
# ================================================================
load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1"
DB_PATH = "master_ai_history.db"

MODELS = {
    "Gemini":    "google/gemini-2.0-flash-exp:free",
    "DeepSeek":  "deepseek/deepseek-chat-v3-0324:free",
    "Llama 3.3": "meta-llama/llama-3.3-70b-instruct:free",
    "Qwen 2.5":  "qwen/qwen-2.5-72b-instruct:free",
    "Mistral":   "mistralai/mistral-small-24b-instruct-2501:free",
}

JUDGE_MODEL = "google/gemini-2.0-flash-exp:free"
IMAGE_MODELS = {
    "Flux Schnell": "black-forest-labs/flux-schnell",
    "DALL-E 3":     "openai/dall-e-3",
}

MAX_TOKENS = 900
TEMPERATURE = 0.7
TIMEOUT = 60
DEBATE_ROUNDS = 2

# Clients
async_client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
sync_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


# ================================================================
# 2. DATABASE — History save karne ke liye
# ================================================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS chats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT, role TEXT, content TEXT,
        mode TEXT, timestamp TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS ai_responses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id INTEGER, model_name TEXT,
        response TEXT, timestamp TEXT)""")
    conn.commit()
    conn.close()

def save_message(session_id, role, content, mode="chat"):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO chats (session_id, role, content, mode, timestamp) "
              "VALUES (?, ?, ?, ?, ?)",
              (session_id, role, content, mode, datetime.now().isoformat()))
    cid = c.lastrowid
    conn.commit()
    conn.close()
    return cid

def save_ai_responses(chat_id, responses):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    for r in responses:
        c.execute("INSERT INTO ai_responses (chat_id, model_name, response, timestamp) "
                  "VALUES (?, ?, ?, ?)",
                  (chat_id, r["name"], r["answer"], datetime.now().isoformat()))
    conn.commit()
    conn.close()

def get_history(session_id, limit=50):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT role, content FROM chats WHERE session_id = ? "
              "ORDER BY id DESC LIMIT ?", (session_id, limit))
    rows = c.fetchall()
    conn.close()
    return list(reversed(rows))

def get_all_sessions():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT session_id, COUNT(*) FROM chats "
              "GROUP BY session_id ORDER BY MIN(timestamp) DESC LIMIT 10")
    rows = c.fetchall()
    conn.close()
    return rows

def clear_session(sid):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM chats WHERE session_id = ?", (sid,))
    conn.commit()
    conn.close()

def clear_all():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM chats")
    conn.execute("DELETE FROM ai_responses")
    conn.commit()
    conn.close()

init_db()


# ================================================================
# 3. CORE AI — Ask models, Judge, Consensus
# ================================================================
async def ask_model(name, model_id, query):
    try:
        resp = await async_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": query}],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            timeout=TIMEOUT,
        )
        return {"name": name, "answer": resp.choices[0].message.content, "status": "ok"}
    except Exception as e:
        return {"name": name, "answer": f"❌ {str(e)[:120]}", "status": "error"}

async def get_all_answers(query):
    tasks = [ask_model(n, m, query) for n, m in MODELS.items()]
    return await asyncio.gather(*tasks)

async def judge_answers(query, results):
    valid = [r for r in results if r["status"] == "ok"]
    if not valid:
        return "❌ सब AI fail हो गए। दोबारा try करें।"

    text = "\n\n".join([f"### {r['name']}:\n{r['answer']}" for r in valid])
    prompt = f"""तुम निष्पक्ष न्यायाधीश हो।

**सवाल:** {query}

**AI के जवाब:**
{text}

**काम:** सबकी तुलना करो, गलती सुधारो, सबसे सही final जवाब दो।

**Format:**
🎯 **सही जवाब:** ...
📊 **कारण:** ...
🔍 **हर AI का योगदान:** (संक्षेप में)
💯 **Confidence:** X%"""

    try:
        resp = await async_client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000, temperature=0.3, timeout=90,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"❌ Judge error: {str(e)[:120]}"

async def master_ai(query):
    results = await get_all_answers(query)
    final = await judge_answers(query, results)
    return {"individual": results, "final": final}


# ================================================================
# 4. DEBATE — AI आपस में बहस करें
# ================================================================
async def debate_round(query, prev_answers, rnd):
    summary = "\n\n".join([f"**{n}:** {a}" for n, a in prev_answers.items()])
    prompt = f"""सवाल: {query}

**पिछले round में:**
{summary}

**Round {rnd}:** दूसरों की गलती पकड़ो, अपना जवाब सुधारो।"""
    tasks = [ask_model(n, m, prompt) for n, m in MODELS.items()]
    results = await asyncio.gather(*tasks)
    return {r["name"]: r["answer"] for r in results if r["status"] == "ok"}

async def run_debate(query, rounds=DEBATE_ROUNDS):
    initial = await get_all_answers(query)
    answers = {r["name"]: r["answer"] for r in initial if r["status"] == "ok"}
    history = [{"round": 0, "answers": answers.copy()}]

    for i in range(1, rounds + 1):
        answers = await debate_round(query, answers, i)
        history.append({"round": i, "answers": answers.copy()})

    final_prompt = f"सवाल: {query}\n\n{rounds} round debate के बाद final जवाब:\n\n"
    for n, a in answers.items():
        final_prompt += f"**{n}:** {a}\n\n"
    final_prompt += "\nConsensus निकालकर final जवाब दो + Confidence %"

    try:
        resp = await async_client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": final_prompt}],
            max_tokens=2000, temperature=0.3, timeout=90,
        )
        final = resp.choices[0].message.content
    except Exception as e:
        final = f"❌ {str(e)[:120]}"

    return {"debate_history": history, "final": final}


# ================================================================
# 5. IMAGE GENERATION
# ================================================================
def enhance_prompt(user_input):
    try:
        resp = sync_client.chat.completions.create(
            model="google/gemini-2.0-flash-exp:free",
            messages=[{"role": "user", "content":
                f"Convert to detailed English image prompt (style, lighting, colors): {user_input}. Output only the prompt."}],
            max_tokens=300,
        )
        return resp.choices[0].message.content.strip()
    except:
        return user_input

def generate_image(prompt, model_key="Flux Schnell"):
    try:
        resp = sync_client.chat.completions.create(
            model=IMAGE_MODELS.get(model_key, IMAGE_MODELS["Flux Schnell"]),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
        )
        msg = resp.choices[0].message
        if hasattr(msg, "images") and msg.images:
            url = msg.images[0].get("image_url", {}).get("url", "")
            if url.startswith("data:image"):
                return {"status": "ok", "image": base64.b64decode(url.split(",")[1])}
            return {"status": "ok", "url": url}
        return {"status": "error", "message": "कोई image नहीं मिली"}
    except Exception as e:
        return {"status": "error", "message": str(e)[:150]}


# ================================================================
# 6. VOICE → TEXT (Whisper)
# ================================================================
def transcribe_audio(audio_bytes):
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            path = tmp.name
        with open(path, "rb") as f:
            resp = sync_client.audio.transcriptions.create(
                model="openai/whisper-1", file=f)
        os.unlink(path)
        return {"status": "ok", "text": resp.text}
    except Exception as e:
        return {"status": "error", "message": str(e)[:150]}


# ================================================================
# 7. STREAMLIT UI
# ================================================================
st.set_page_config(page_title="Master AI", page_icon="🧠", layout="wide")

st.markdown("""
<style>
.big-title { font-size: 2.4rem; font-weight: bold;
    background: linear-gradient(90deg, #667eea, #764ba2);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.badge { padding: 3px 10px; border-radius: 10px; background: #e3f2fd;
    color: #1565c0; font-size: 0.8rem; margin: 2px; display: inline-block; }
</style>
""", unsafe_allow_html=True)

# Session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- Sidebar ----
with st.sidebar:
    st.markdown("### 🧠 Master AI")
    st.caption(f"Session: `{st.session_state.session_id}`")
    st.divider()

    mode = st.radio("**🎯 Mode:**",
        ["💬 Chat (Consensus)", "🥊 Debate", "🎨 Image Generation"],
        label_visibility="collapsed")

    st.divider()
    st.markdown("**🤖 Models:**")
    for name in MODELS.keys():
        st.markdown(f'<span class="badge">• {name}</span>', unsafe_allow_html=True)

    st.divider()
    voice_on = st.toggle("🎤 Voice Input", value=False)

    st.divider()
    with st.expander("📜 पुरानी चैट"):
        sessions = get_all_sessions()
        if sessions:
            for sid, cnt in sessions:
                c1, c2 = st.columns([3, 1])
                c1.caption(f"`{sid}` • {cnt} msgs")
                if c2.button("🗑️", key=f"d_{sid}"):
                    clear_session(sid); st.rerun()
        else:
            st.caption("खाली है")

    if st.button("🧹 Clear Current"):
        st.session_state.messages = []; st.rerun()
    if st.button("⚠️ Delete All"):
        clear_all(); st.session_state.messages = []; st.rerun()

# ---- Header ----
st.markdown('<div class="big-title">🧠 Master AI</div>', unsafe_allow_html=True)
st.caption("Multi-AI · Consensus · Debate · Image · Voice")

# Load history
if not st.session_state.messages:
    for role, content in get_history(st.session_state.session_id):
        st.session_state.messages.append({"role": role, "content": content})

# Display history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ---- Voice Input ----
user_input = None
if voice_on:
    try:
        from streamlit_mic_recorder import mic_recorder
        audio = mic_recorder(start_prompt="🎤 बोलें", stop_prompt="⏹️ रोकें",
                             just_once=True, key="mic")
        if audio and audio.get("bytes"):
            with st.spinner("🎧 सुन रहे हैं..."):
                r = transcribe_audio(audio["bytes"])
            if r["status"] == "ok":
                user_input = r["text"]
                st.success(f"सुना: {user_input}")
            else:
                st.error(r["message"])
    except ImportError:
        st.warning("`streamlit-mic-recorder` install करें")

# ---- Text Input ----
prompt = st.chat_input("कुछ भी पूछें...")
if prompt:
    user_input = prompt

# ---- Process ----
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    save_message(st.session_state.session_id, "user", user_input, mode)
    with st.chat_message("user"):
        st.markdown(user_input)

    # CHAT
    if "Chat" in mode:
        with st.chat_message("assistant"):
            with st.spinner("🤖 सब AI से जवाब ला रहे हैं..."):
                result = asyncio.run(master_ai(user_input))
            cid = save_message(st.session_state.session_id, "assistant",
                               result["final"], "chat")
            save_ai_responses(cid, result["individual"])

            with st.expander("🔍 सब AI का जवाब देखें"):
                for r in result["individual"]:
                    icon = "✅" if r["status"] == "ok" else "❌"
                    st.markdown(f"**{icon} {r['name']}**")
                    st.markdown(r["answer"])
                    st.divider()

            st.markdown("### ⚖️ Judge का अंतिम निर्णय")
            st.markdown(result["final"])
            st.session_state.messages.append(
                {"role": "assistant", "content": result["final"]})

    # DEBATE
    elif "Debate" in mode:
        with st.chat_message("assistant"):
            prog = st.progress(0, text="🥊 Debate शुरू...")
            result = asyncio.run(run_debate(user_input, DEBATE_ROUNDS))
            prog.progress(100, text="✅ पूरा")

            with st.expander("🥊 पूरी बहस देखें"):
                for e in result["debate_history"]:
                    st.markdown(f"### Round {e['round']}")
                    for n, a in e["answers"].items():
                        st.markdown(f"**{n}:** {a}")
                        st.divider()

            st.markdown("### ⚖️ Debate का निर्णय")
            st.markdown(result["final"])
            save_message(st.session_state.session_id, "assistant",
                         result["final"], "debate")
            st.session_state.messages.append(
                {"role": "assistant", "content": result["final"]})

    # IMAGE
    elif "Image" in mode:
        with st.chat_message("assistant"):
            with st.spinner("🎨 Prompt enhance..."):
                enh = enhance_prompt(user_input)
            st.caption(f"Enhanced: *{enh}*")

            with st.spinner("🖼️ Image बना रहे हैं..."):
                img = generate_image(enh)

            if img["status"] == "ok":
                if img.get("url"):
                    st.image(img["url"], caption=user_input)
                    resp_text = f"🖼️ [Download]({img['url']})"
                else:
                    st.image(img["image"], caption=user_input)
                    resp_text = f"🖼️ Image: {user_input}"
            else:
                resp_text = f"❌ {img['message']}"
                st.error(resp_text)

            save_message(st.session_state.session_id, "assistant",
                         resp_text, "image")
            st.session_state.messages.append(
                {"role": "assistant", "content": resp_text})
