# ================================================================
# 🧠 Master AI — Supreme Court System (Final)
# 7 AI + 3 Round Debate + Claude Judge + Cross-Verification
# ================================================================

import os
import asyncio
import streamlit as st
from openai import AsyncOpenAI
from google import genai


# ================================================================
# 1. API KEYS (कहीं missing हो तो चलता रहे)
# ================================================================
def get_secret(k):
    try:
        return st.secrets.get(k)
    except Exception:
        return os.getenv(k)

OPENROUTER_KEY = get_secret("OPENROUTER_API_KEY")   # ⭐ मुख्य key
GEMINI_KEY     = get_secret("GEMINI_API_KEY")
OPENAI_KEY     = get_secret("OPENAI_API_KEY")
ANTHROPIC_KEY  = get_secret("ANTHROPIC_API_KEY")
DEEPSEEK_KEY   = get_secret("DEEPSEEK_API_KEY")
PERPLEXITY_KEY = get_secret("PERPLEXITY_API_KEY")
GROK_KEY       = get_secret("GROK_API_KEY")

OR_BASE = "https://openrouter.ai/api/v1"


# ================================================================
# 2. हर AI का function — पहले OpenRouter से, fail हो तो direct
# ================================================================
async def _ask_openrouter(model_id, prompt, name):
    """OpenRouter के जरिए किसी भी model से जवाब"""
    try:
        client = AsyncOpenAI(api_key=OPENROUTER_KEY, base_url=OR_BASE)
        resp = await client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            timeout=90,
        )
        return {"name": name, "answer": resp.choices[0].message.content, "status": "ok"}
    except Exception as e:
        return {"name": name, "answer": f"❌ {str(e)[:150]}", "status": "error"}


# --- Gemini (OpenRouter via) ---
async def ask_gemini(prompt):
    # पहले OpenRouter से try, fail हो तो direct Google
    r = await _ask_openrouter("google/gemini-2.5-flash", prompt, "Gemini")
    if r["status"] == "ok":
        return r
    if GEMINI_KEY:
        try:
            client = genai.Client(api_key=GEMINI_KEY)
            resp = await client.aio.models.generate_content(
                model="gemini-2.5-flash", contents=prompt,
            )
            return {"name": "Gemini", "answer": resp.text, "status": "ok"}
        except Exception as e:
            return {"name": "Gemini", "answer": f"❌ {str(e)[:120]}", "status": "error"}
    return r


# --- ChatGPT ---
async def ask_openai(prompt):
    r = await _ask_openrouter("openai/gpt-4o-mini", prompt, "ChatGPT")
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


# --- Claude ---
async def ask_claude(prompt):
    r = await _ask_openrouter("anthropic/claude-3.5-sonnet", prompt, "Claude")
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


# --- DeepSeek ---
async def ask_deepseek(prompt):
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


# --- Perplexity ---
async def ask_perplexity(prompt):
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


# --- Grok ---
async def ask_grok(prompt):
    r = await _ask_openrouter("x-ai/grok-2-latest", prompt, "Grok")
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


# --- OpenRouter Free Router (7वां AI) ---
async def ask_openrouter_free(prompt):
    return await _ask_openrouter("openrouter/free", prompt, "OpenRouter")


# ================================================================
# 3. ROUND 1 — सब AI से पहला जवाब
# ================================================================
async def round_1_initial(query):
    tasks = [
        ask_gemini(query),
        ask_openai(query),
        ask_claude(query),
        ask_deepseek(query),
        ask_perplexity(query),
        ask_grok(query),
        ask_openrouter_free(query),
    ]
    return await asyncio.gather(*tasks)


# ================================================================
# 4. ROUND 2-3 — AI आपस में बहस करें
# ================================================================
async def debate_round(query, prev_answers, round_num):
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
        ask_gemini(debate_prompt),
        ask_openai(debate_prompt),
        ask_claude(debate_prompt),
        ask_deepseek(debate_prompt),
        ask_perplexity(debate_prompt),
        ask_grok(debate_prompt),
        ask_openrouter_free(debate_prompt),
    ]
    return await asyncio.gather(*tasks)


# ================================================================
# 5. SUPREME JUDGE — Claude ही मुख्य Judge
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

**चरण 2 — Cross-Verification:** कौन-कौन से AI एक ही बात कह रहे हैं? कहाँ टकराव है?

**चरण 3 — Counter-Argument:** हर मुख्य जवाब के खिलाफ एक विरोधी तर्क बनाओ।

**चरण 4 — Evidence:** किस जवाब के पीछे तथ्य/आंकड़ा/स्रोत है?

**चरण 5 — Weak Points:** किस जवाब में कमी या शक है — उजागर करो।

**चरण 6 — Consensus:** बहुमत किस बात पर सहमत है?

**चरण 7 — FINAL VERDICT:** वह जवाब दो जो सारी जाँच में टिक गया।

============================================================
Output Format (हिंदी में):
============================================================

🎯 **अंतिम सही जवाब (Final Verdict):**
[सबसे सही, पूरा, स्पष्ट और पक्का जवाब]

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

💯 **Confidence:** [X]% — [कारण]

🛡️ **क्या यह जवाब काटा जा सकता है?** [हाँ/नहीं — कारण]
"""

    # ⭐ Judge: Claude मुख्य, बाकी fallback
    judges = [
        ("Claude (OpenRouter)", OPENROUTER_KEY, OR_BASE, "anthropic/claude-3.5-sonnet"),
        ("Claude (Direct)", ANTHROPIC_KEY, "https://api.anthropic.com/v1", "claude-3-5-sonnet-20241022"),
        ("ChatGPT", OPENROUTER_KEY, OR_BASE, "openai/gpt-4o-mini"),
        ("DeepSeek", OPENROUTER_KEY, OR_BASE, "deepseek/deepseek-chat"),
        ("Grok", OPENROUTER_KEY, OR_BASE, "x-ai/grok-2-latest"),
    ]

    for judge_name, key, base_url, model in judges:
        if not key:
            continue
        try:
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
# 6. पूरा Master Flow
# ================================================================
async def master_ai(query, total_rounds=3):
    all_rounds = []

    r1 = await round_1_initial(query)
    all_rounds.append({"round": 1, "answers": r1})

    for i in range(2, total_rounds + 1):
        prev = all_rounds[-1]["answers"]
        rN = await debate_round(query, prev, i)
        all_rounds.append({"round": i, "answers": rN})

    final = await supreme_judge(query, all_rounds)
    return {"rounds": all_rounds, "final": final}


# ================================================================
# 7. Streamlit UI
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

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown("### 🧠 Master AI")
    st.caption("Supreme Court System")
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
    if not OPENROUTER_KEY:
        st.error("⚠️ OPENROUTER_API_KEY नहीं मिली")
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

st.markdown('<div class="big-title">🧠 Master AI — Supreme Court</div>', unsafe_allow_html=True)
st.caption("7 AI · 3 Round Debate · Cross-Verification · Claude Judge")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("कुछ भी पूछें...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        prog = st.progress(0, text="🤖 7 AI जवाब दे रहे हैं...")
        result = asyncio.run(master_ai(prompt, total_rounds=rounds))
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

        st.session_state.messages.append(
            {"role": "assistant", "content": result["final"]}
        )
