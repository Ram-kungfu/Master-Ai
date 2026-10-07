# ================================================================
# 🧠 Master AI — Supreme Court System (Full Final + Gemini Judge)
# 7 AI + 3 Round + Claude Judge + Gemini Fallback + History + Image + PDF + Voice
# ================================================================

import os
import uuid
import base64
import sqlite3
import asyncio
from datetime import datetime

import streamlit as st
from openai import AsyncOpenAI
from google import genai


# ================================================================
# 1. API KEYS
# ================================================================
def get_secret(k):
    try:
        return st.secrets.get(k)
    except Exception:
        return os.getenv(k)

OPENROUTER_KEY = get_secret("OPENROUTER_API_KEY")
GEMINI_KEY     = get_secret("GEMINI_API_KEY")
OPENAI_KEY     = get_secret("OPENAI_API_KEY")
ANTHROPIC_KEY  = get_secret("ANTHROPIC_API_KEY")
DEEPSEEK_KEY   = get_secret("DEEPSEEK_API_KEY")
PERPLEXITY_KEY = get_secret("PERPLEXITY_API_KEY")
GROK_KEY       = get_secret("GROK_API_KEY")
("OpenRouter Free", OPENROUTER_KEY, OR_BASE, "openrouter/free"),

OR_BASE = "https://openrouter.ai/api/v1"
DB_PATH = "master_ai_history.db"


# ================================================================
# 2. DATABASE
# ================================================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS chats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        role TEXT,
        content TEXT,
        timestamp TEXT
    )""")
    conn.commit()
    conn.close()


def save_message(session_id, role, content):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO chats (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (session_id, role, content, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def load_history(session_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT role, content FROM chats WHERE session_id = ? ORDER BY id ASC",
        (session_id,)
    )
    rows = c.fetchall()
    conn.close()
    fixed = []
    for row in rows:
        if isinstance(row, (list, tuple)) and len(row) >= 2:
            fixed.append({"role": row[0], "content": row[1]})
        elif isinstance(row, dict):
            fixed.append(row)
    return fixed


def get_all_sessions():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""SELECT session_id, MIN(timestamp), COUNT(*)
                 FROM chats GROUP BY session_id
                 ORDER BY MIN(timestamp) DESC LIMIT 20""")
    rows = c.fetchall()
    conn.close()
    return rows


def delete_session(session_id):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM chats WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()


init_db()


# ================================================================
# 3. AI FUNCTIONS — 7 AI
# ================================================================
async def _ask_openrouter(model_id, prompt, name, image_b64=None):
    try:
        client = AsyncOpenAI(api_key=OPENROUTER_KEY, base_url=OR_BASE)
        if image_b64:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
            }]
        else:
            messages = [{"role": "user", "content": prompt}]

        resp = await client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=2000,
            timeout=90,
        )
        return {"name": name, "answer": resp.choices[0].message.content, "status": "ok"}
    except Exception as e:
        return {"name": name, "answer": f"❌ {str(e)[:150]}", "status": "error"}


async def ask_gemini(prompt, image_b64=None):
    r = await _ask_openrouter("google/gemini-2.5-flash", prompt, "Gemini", image_b64)
    if r["status"] == "ok":
        return r
    if GEMINI_KEY:
        try:
            client = genai.Client(api_key=GEMINI_KEY)
            contents = [prompt]
            if image_b64:
                contents.append({
                    "mime_type": "image/jpeg",
                    "data": base64.b64decode(image_b64)
                })
            resp = await client.aio.models.generate_content(
                model="gemini-2.5-flash", contents=contents,
            )
            return {"name": "Gemini", "answer": resp.text, "status": "ok"}
        except Exception as e:
            return {"name": "Gemini", "answer": f"❌ {str(e)[:120]}", "status": "error"}
    return r


async def ask_openai(prompt, image_b64=None):
    r = await _ask_openrouter("openai/gpt-4o-mini", prompt, "ChatGPT", image_b64)
    if r["status"] == "ok":
        return r
    if OPENAI_KEY:
        try:
            client = AsyncOpenAI(api_key=OPENAI_KEY)
            resp = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000, timeout=90,
            )
            return {"name": "ChatGPT", "answer": resp.choices[0].message.content, "status": "ok"}
        except Exception as e:
            return {"name": "ChatGPT", "answer": f"❌ {str(e)[:120]}", "status": "error"}
    return r


async def ask_claude(prompt, image_b64=None):
    r = await _ask_openrouter("anthropic/claude-3.5-sonnet", prompt, "Claude", image_b64)
    if r["status"] == "ok":
        return r
    if ANTHROPIC_KEY:
        try:
            client = AsyncOpenAI(api_key=ANTHROPIC_KEY, base_url="https://api.anthropic.com/v1")
            resp = await client.chat.completions.create(
                model="claude-3-5-sonnet-20241022",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000, timeout=90,
            )
            return {"name": "Claude", "answer": resp.choices[0].message.content, "status": "ok"}
        except Exception as e:
            return {"name": "Claude", "answer": f"❌ {str(e)[:120]}", "status": "error"}
    return r


async def ask_deepseek(prompt, image_b64=None):
    r = await _ask_openrouter("deepseek/deepseek-chat", prompt, "DeepSeek")
    if r["status"] == "ok":
        return r
    if DEEPSEEK_KEY:
        try:
            client = AsyncOpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com")
            resp = await client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000, timeout=90,
            )
            return {"name": "DeepSeek", "answer": resp.choices[0].message.content, "status": "ok"}
        except Exception as e:
            return {"name": "DeepSeek", "answer": f"❌ {str(e)[:120]}", "status": "error"}
    return r


async def ask_perplexity(prompt, image_b64=None):
    r = await _ask_openrouter("perplexity/sonar-pro", prompt, "Perplexity")
    if r["status"] == "ok":
        return r
    if PERPLEXITY_KEY:
        try:
            client = AsyncOpenAI(api_key=PERPLEXITY_KEY, base_url="https://api.perplexity.ai")
            resp = await client.chat.completions.create(
                model="sonar-pro",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000, timeout=90,
            )
            return {"name": "Perplexity", "answer": resp.choices[0].message.content, "status": "ok"}
        except Exception as e:
            return {"name": "Perplexity", "answer": f"❌ {str(e)[:120]}", "status": "error"}
    return r


async def ask_grok(prompt, image_b64=None):
    r = await _ask_openrouter("x-ai/grok-2-vision-1212", prompt, "Grok", image_b64)
    if r["status"] == "ok":
        return r
    if GROK_KEY:
        try:
            client = AsyncOpenAI(api_key=GROK_KEY, base_url="https://api.x.ai/v1")
            resp = await client.chat.completions.create(
                model="grok-2-latest",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000, timeout=90,
            )
            return {"name": "Grok", "answer": resp.choices[0].message.content, "status": "ok"}
        except Exception as e:
            return {"name": "Grok", "answer": f"❌ {str(e)[:120]}", "status": "error"}
    return r


async def ask_openrouter_free(prompt, image_b64=None):
    return await _ask_openrouter("openrouter/free", prompt, "OpenRouter")


# ================================================================
# 4. ROUND 1 — सब AI से पहला जवाब
# ================================================================
async def round_1_initial(query, image_b64=None):
    tasks = [
        ask_gemini(query, image_b64),
        ask_openai(query, image_b64),
        ask_claude(query, image_b64),
        ask_deepseek(query),
        ask_perplexity(query),
        ask_grok(query, image_b64),
        ask_openrouter_free(query),
    ]
    return await asyncio.gather(*tasks)


# ================================================================
# 5. ROUND 2-3 — AI आपस में बहस करें
# ================================================================
async def debate_round(query, prev_answers, round_num, image_b64=None):
    summary = "\n\n".join([
        f"**{a['name']}:**\n{a['answer']}"
        for a in prev_answers if a["status"] == "ok"
    ])

    debate_prompt = f"""तुम एक बुद्धिमान, तर्कशील AI हो।

**User का सवाल:** {query}

**पिछले round में बाकी AI ने यह कहा:**
{summary}

**Round {round_num} — तुम्हारा काम (सख्ती से करो):**

1. **दूसरों के जवाब पढ़ो** — कहाँ सही हैं, कहाँ गलत
2. **Counter-argument बनाओ** — अगर किसी का जवाब उल्टा साबित हो तो कैसे?
3. **Evidence माँगो** — किस तथ्य, आँकड़े या स्रोत से पक्का होता है?
4. **Weak points पकड़ो** — किसी की बात में कमी है तो उजागर करो
5. **अपना सुधारा हुआ जवाब दो** — सबसे मजबूत तर्क के साथ

**सिर्फ अपना updated जवाब लिखो — भूमिका नहीं, बहाना नहीं।**
"""

    tasks = [
        ask_gemini(debate_prompt, image_b64),
        ask_openai(debate_prompt),
        ask_claude(debate_prompt),
        ask_deepseek(debate_prompt),
        ask_perplexity(debate_prompt),
        ask_grok(debate_prompt),
        ask_openrouter_free(debate_prompt),
    ]
    return await asyncio.gather(*tasks)


# ================================================================
# 6. SUPREME JUDGE — Claude पहला Judge + Gemini Fallback
# ================================================================
async def supreme_judge(query, all_rounds):
    summary_parts = []
    for rnd in all_rounds:
        rn = rnd["round"]
        valid = [a for a in rnd["answers"] if a["status"] == "ok"]
        text = "\n\n".join([f"**{a['name']}:**\n{a['answer']}" for a in valid])
        summary_parts.append(f"### Round {rn}\n{text}")

    full_debate = "\n\n---\n\n".join(summary_parts)

    judge_prompt = f"""तुम Supreme Court के मुख्य न्यायाधीश हो।
नीचे {len(all_rounds)} round की पूरी बहस है — कई AI ने हिस्सा लिया।

**User का सवाल:**
{query}

**पूरी बहस:**
{full_debate}

============================================================
तुम्हारा काम — 7 सख्त चरणों में:
============================================================

**चरण 1 — सबको पढ़ो:** हर round के हर AI का जवाब ध्यान से पढ़ो।
**चरण 2 — Cross-Verification:** कौन-कौन से AI एक ही बात कह रहे हैं?
**चरण 3 — Counter-Argument:** हर मुख्य जवाब के खिलाफ विरोधी तर्क बनाओ।
**चरण 4 — Evidence:** किस जवाब के पीछे तथ्य/आंकड़ा/स्रोत है?
**चरण 5 — Weak Points:** किस जवाब में कमी या शक है।
**चरण 6 — Consensus:** बहुमत किस बात पर सहमत है?
**चरण 7 — FINAL VERDICT:** वह जवाब दो जो सारी जाँच में टिक गया।

============================================================
Output Format (हिंदी में):
============================================================

🎯 **अंतिम सही जवाब (Final Verdict):**
[सबसे सही, पूरा, स्पष्ट जवाब]

🧪 **Counter-Arguments की जाँच:**
[कौन-कौन से विरोधी तर्क उठाए गए]

📊 **Evidence (सबूत):**
[किस तथ्य/स्रोत के आधार पर सही है]

🤝 **Consensus (सहमति):**
[कितने AI एक ही बात पर आए]

⚠️ **Weak Points:**
[कहाँ अभी भी शक बाकी है]

🔍 **हर AI का योगदान:**
- Gemini: [क्या कहा]
- ChatGPT: [क्या कहा]
- Claude: [क्या कहा]
- DeepSeek: [क्या कहा]
- Perplexity: [क्या कहा]
- Grok: [क्या कहा]
- OpenRouter: [क्या कहा]

💯 **Confidence:** [X]%

🛡️ **क्या यह जवाब काटा जा सकता है?** [हाँ/नहीं]
"""

    # ⭐ Claude पहला Judge — Gemini fallback
    judges = [
        ("Claude (OpenRouter)", OPENROUTER_KEY, OR_BASE, "anthropic/claude-3.5-sonnet"),
        ("Claude (Direct)", ANTHROPIC_KEY, "https://api.anthropic.com/v1", "claude-3-5-sonnet-20241022"),
        ("ChatGPT", OPENROUTER_KEY, OR_BASE, "openai/gpt-4o-mini"),
        ("DeepSeek", OPENROUTER_KEY, OR_BASE, "deepseek/deepseek-chat"),
        ("Grok", OPENROUTER_KEY, OR_BASE, "x-ai/grok-2-latest"),
        ("Gemini", GEMINI_KEY, None, "gemini-2.5-flash"),
    ]

    for judge_name, key, base_url, model in judges:
        if not key:
            continue
        try:
            # Gemini के लिए genai
            if judge_name == "Gemini":
                client = genai.Client(api_key=key)
                resp = await client.aio.models.generate_content(
                    model=model, contents=judge_prompt,
                )
                return f"**Judge: {judge_name}**\n\n" + resp.text
            else:
                # बाकी सब OpenAI-compatible
                client = AsyncOpenAI(api_key=key, base_url=base_url)
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": judge_prompt}],
                    max_tokens=5000,
                    timeout=180,
                )
                return f"**Judge: {judge_name}**\n\n" + resp.choices[0].message.content
        except Exception:
            continue

    return "❌ सब Judge fail हो गए।"


# ================================================================
# 7. MASTER FLOW
# ================================================================
async def master_ai(query, total_rounds=3, image_b64=None):
    all_rounds = []
    r1 = await round_1_initial(query, image_b64)
    all_rounds.append({"round": 1, "answers": r1})

    for i in range(2, total_rounds + 1):
        prev = all_rounds[-1]["answers"]
        rN = await debate_round(query, prev, i, image_b64)
        all_rounds.append({"round": i, "answers": rN})

    final = await supreme_judge(query, all_rounds)
    return {"rounds": all_rounds, "final": final}


# ================================================================
# 8. STREAMLIT UI
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

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "messages" not in st.session_state:
    st.session_state.messages = []


# ---- Sidebar ----
with st.sidebar:
    st.markdown("### 🧠 Master AI")
    st.caption(f"Session: `{st.session_state.session_id}`")
    st.divider()

    st.markdown("**🤖 7 AI:**")
    for n in ["Gemini", "ChatGPT", "Claude", "DeepSeek", "Perplexity", "Grok", "OpenRouter"]:
        st.markdown(f'<span class="badge">• {n}</span>', unsafe_allow_html=True)

    st.divider()
    st.markdown("**⚖️ Judge:** Claude 3.5 Sonnet")
    st.divider()

    rounds = st.slider("Debate Rounds", 1, 5, 3,
                       help="3 = बेहतर | 4-5 = गहरे सवालों के लिए")

    st.divider()
    st.markdown("**📷 Image Upload:**")
    uploaded_image = st.file_uploader(
        "Image चुनें",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("**📄 PDF Upload:**")
    uploaded_pdf = st.file_uploader(
        "PDF चुनें",
        type=["pdf"],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("**🎤 Voice Input:**")
    voice_on = st.toggle("Voice ON", value=False)

    st.divider()
    st.markdown("**💾 History:**")
    sessions = get_all_sessions()
    if sessions:
        for sid, ts, cnt in sessions[:10]:
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button(f"📄 {sid} ({cnt})", key=f"load_{sid}"):
                    st.session_state.session_id = sid
                    st.session_state.messages = load_history(sid)
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{sid}"):
                    delete_session(sid)
                    if st.session_state.session_id == sid:
                        st.session_state.session_id = str(uuid.uuid4())[:8]
                        st.session_state.messages = []
                    st.rerun()
    else:
        st.caption("कोई history नहीं")

    st.divider()
    if st.button("🧹 New Chat"):
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.session_state.messages = []
        st.rerun()

    # ✅ Download — हमेशा दिखे (chat खाली हो तो भी)
    if st.session_state.messages and len(st.session_state.messages) > 0:
        parts = []
        for m in st.session_state.messages:
            if isinstance(m, dict):
                role = m.get("role", "user")
                content = m.get("content", "")
            elif isinstance(m, (list, tuple)) and len(m) >= 2:
                role, content = m[0], m[1]
            else:
                continue
            parts.append(f"**{str(role).upper()}:**\n{content}")
        chat_text = "\n\n".join(parts)
    else:
        chat_text = "अभी कोई chat नहीं — पहले सवाल पूछें।"

    st.download_button(
        "📥 Download Chat",
        data=chat_text,
        file_name=f"master_ai_{st.session_state.session_id}.md",
        mime="text/markdown"
    )

    st.divider()
    if not OPENROUTER_KEY:
        st.error("⚠️ OPENROUTER_API_KEY नहीं मिली")


# ---- Main ----
st.markdown('<div class="big-title">🧠 Master AI — Supreme Court</div>', unsafe_allow_html=True)
st.caption("7 AI · 3 Round Debate · Judge · History · Image · PDF · Voice")


if not st.session_state.messages:
    st.session_state.messages = load_history(st.session_state.session_id)


for m in st.session_state.messages:
    if isinstance(m, dict):
        role = m.get("role", "user")
        content = m.get("content", "")
    elif isinstance(m, (list, tuple)) and len(m) >= 2:
        role, content = m[0], m[1]
    else:
        continue

    with st.chat_message(role):
        st.markdown(content)
        if role == "assistant":
            with st.expander("📋 Copy जवाब"):
                st.code(content, language=None)


image_b64 = None
if uploaded_image:
    st.image(uploaded_image, caption="Uploaded Image", width=300)
    image_b64 = base64.b64encode(uploaded_image.read()).decode("utf-8")

pdf_text = ""
if uploaded_pdf:
    try:
        pdf_bytes = uploaded_pdf.read()
        pdf_text = f"[PDF: {uploaded_pdf.name}, {len(pdf_bytes)} bytes]"
        st.info(f"📄 PDF loaded: {uploaded_pdf.name}")
    except Exception as e:
        st.warning(f"PDF load error: {e}")


user_input_voice = None
if voice_on:
    try:
        from streamlit_mic_recorder import mic_recorder
        audio = mic_recorder(
            start_prompt="🎤 बोलें",
            stop_prompt="⏹️ रोकें",
            just_once=True,
            key="mic"
        )
        if audio and audio.get("bytes"):
            user_input_voice = "🎤 [Voice input detected]"
    except ImportError:
        st.warning("Voice के लिए `streamlit-mic-recorder` install करें")


prompt = st.chat_input("कुछ भी पूछें...")

final_prompt = None
if prompt:
    final_prompt = prompt
elif user_input_voice:
    final_prompt = user_input_voice

if final_prompt:
    display_msg = final_prompt
    if uploaded_image:
        display_msg = f"📷 [Image attached]\n\n{final_prompt}"
    if uploaded_pdf:
        display_msg = f"📄 [PDF attached]\n\n{display_msg}"

    st.session_state.messages.append({"role": "user", "content": display_msg})
    save_message(st.session_state.session_id, "user", display_msg)

    with st.chat_message("user"):
        st.markdown(display_msg)

    full_query = final_prompt
    if pdf_text:
        full_query = f"{pdf_text}\n\n{final_prompt}"

    with st.chat_message("assistant"):
        prog = st.progress(0, text="🤖 7 AI जवाब दे रहे हैं...")
        result = asyncio.run(master_ai(full_query, total_rounds=rounds, image_b64=image_b64))
        prog.progress(100, text="✅ पूरा")

        for rnd in result["rounds"]:
            with st.expander(f"🥊 Round {rnd['round']} — पूरी बहस", expanded=False):
                for a in rnd["answers"]:
                    icon = "✅" if a["status"] == "ok" else "❌"
                    st.markdown(f"**{icon} {a['name']}**")
                    st.markdown(a["answer"])
                    st.divider()

        st.markdown("## ⚖️ Supreme Judge का अंतिम फैसला")
        st.markdown(result["final"])

        with st.expander("📋 Copy जवाब"):
            st.code(result["final"], language=None)

        st.info("💡 Sidebar से Download भी कर सकते हैं")

        st.session_state.messages.append(
            {"role": "assistant", "content": result["final"]}
        )
        save_message(st.session_state.session_id, "assistant", result["final"])
