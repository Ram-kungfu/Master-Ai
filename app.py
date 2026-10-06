# ================================================================
# 🧠 Master AI — Supreme Court System v2
# 7 AI + 3 Round Debate + Strict Judge + Cross-Verification
# ================================================================

import os
import asyncio
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

GEMINI_KEY     = get_secret("GEMINI_API_KEY")
OPENAI_KEY     = get_secret("OPENAI_API_KEY")
ANTHROPIC_KEY  = get_secret("ANTHROPIC_API_KEY")
DEEPSEEK_KEY   = get_secret("DEEPSEEK_API_KEY")
PERPLEXITY_KEY = get_secret("PERPLEXITY_API_KEY")
GROK_KEY       = get_secret("GROK_API_KEY")
OPENROUTER_KEY = get_secret("OPENROUTER_API_KEY")


# ================================================================
# 2. हर AI से सवाल पूछने वाले functions
# ================================================================
async def ask_gemini(prompt):
    try:
        client = genai.Client(api_key=GEMINI_KEY)
        resp = await client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return {"name": "Gemini", "answer": resp.text, "status": "ok"}
    except Exception as e:
        return {"name": "Gemini", "answer": f"❌ {str(e)[:120]}", "status": "error"}


async def ask_openai(prompt):
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


async def ask_claude(prompt):
    try:
        client = AsyncOpenAI(
            api_key=ANTHROPIC_KEY,
            base_url="https://api.anthropic.com/v1",
        )
        resp = await client.chat.completions.create(
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000, timeout=90,
        )
        return {"name": "Claude", "answer": resp.choices[0].message.content, "status": "ok"}
    except Exception as e:
        return {"name": "Claude", "answer": f"❌ {str(e)[:120]}", "status": "error"}


async def ask_deepseek(prompt):
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


async def ask_perplexity(prompt):
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


async def ask_grok(prompt):
    try:
        client = AsyncOpenAI(
            api_key=GROK_KEY,
            base_url="https://api.x.ai/v1",
        )
        resp = await client.chat.completions.create(
            model="grok-2-latest",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000, timeout=90,
        )
        return {"name": "Grok", "answer": resp.choices[0].message.content, "status": "ok"}
    except Exception as e:
        return {"name": "Grok", "answer": f"❌ {str(e)[:120]}", "status": "error"}


async def ask_openrouter_free(prompt):
    try:
        client = AsyncOpenAI(
            api_key=OPENROUTER_KEY,
            base_url="https://openrouter.ai/api/v1",
        )
        resp = await client.chat.completions.create(
            model="openrouter/free",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000, timeout=90,
        )
        return {"name": "OpenRouter", "answer": resp.choices[0].message.content, "status": "ok"}
    except Exception as e:
        return {"name": "OpenRouter", "answer": f"❌ {str(e)[:120]}", "status": "error"}


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
# 4. ROUND 2-3 — AI आपस में बहस करें, counter-argument बनाएं
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
# 5. SUPREME JUDGE — हर जवाब पर सख्त जाँच
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
तुम्हारा काम — 7 सख्त चरणों में (कोई चरण न छोड़ो):
============================================================

**चरण 1 — सबको पढ़ो:** हर round के हर AI का जवाब ध्यान से पढ़ो।

**चरण 2 — Cross-Verification:** कौन-कौन से AI एक ही बात कह रहे हैं? कहाँ टकराव है?

**चरण 3 — Counter-Argument:** हर मुख्य जवाब के खिलाफ एक विरोधी तर्क बनाओ। "अगर यह गलत हुआ तो?"

**चरण 4 — Evidence माँगो:** किस जवाब के पीछे तथ्य/आंकड़ा/स्रोत है? जिसके पीछे सबूत नहीं — उसे कमजोर मानो।

**चरण 5 — Weak Points:** किस जवाब में कमी, अधूरापन, या शक है — उजागर करो।

**चरण 6 — Consensus:** बहुमत किस बात पर सहमत है? अल्पमत क्या कह रहा है?

**चरण 7 — FINAL VERDICT:** वह जवाब दो जो सारी जाँच में टिक गया। ऐसा जवाब जिस पर कोई उंगली न उठा सके।

============================================================
Output Format (हिंदी में, साफ-साफ):
============================================================

🎯 **अंतिम सही जवाब (Final Verdict):**
[सबसे सही, पूरा, स्पष्ट और पक्का जवाब — सबूत के साथ]

🧪 **Counter-Arguments की जाँच:**
[कौन-कौन से विरोधी तर्क उठाए गए, और वे कैसे खारिज हुए]

📊 **Evidence (सबूत):**
[किस तथ्य, आंकड़े या स्रोत के आधार पर यह जवाब सही है]

🤝 **Consensus (सहमति):**
[कितने AI एक ही बात पर आए — नाम के साथ]

⚠️ **Weak Points (कमजोरियाँ):**
[कहाँ अभी भी शक बाकी है — पूरी ईमानदारी से]

🔍 **हर AI का योगदान:**
- Gemini: [क्या कहा]
- ChatGPT: [क्या कहा]
- Claude: [क्या कहा]
- DeepSeek: [क्या कहा]
- Perplexity: [क्या कहा]
- Grok: [क्या कहा]
- OpenRouter: [क्या कहा]

💯 **Confidence:** [X]% — [क्यों इतना? बाकी शक क्यों?]

🛡️ **क्या यह जवाब काटा जा सकता है?** [हाँ/नहीं — कारण सहित]
"""

    # Judge chain: Claude → GPT-4o → DeepSeek → Gemini
    judges = [
        ("Claude", ANTHROPIC_KEY, "https://api.anthropic.com/v1", "claude-3-5-sonnet-20241022"),
        ("ChatGPT", OPENAI_KEY, None, "gpt-4o-mini"),
        ("DeepSeek", DEEPSEEK_KEY, "https://api.deepseek.com", "deepseek-chat"),
        ("Grok", GROK_KEY, "https://api.x.ai/v1", "grok-2-latest"),
    ]

    for judge_name, key, base_url, model in judges:
        try:
            client = AsyncOpenAI(api_key=key, base_url=base_url) if base_url else AsyncOpenAI(api_key=key)
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

    # Round 1 — पहला जवाब
    r1 = await round_1_initial(query)
    all_rounds.append({"round": 1, "answers": r1})

    # Round 2-N — बहस
    for i in range(2, total_rounds + 1):
        prev = all_rounds[-1]["answers"]
        rN = await debate_round(query, prev, i)
        all_rounds.append({"round": i, "answers": rN})

    # Final Supreme Judge
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
    st.caption("Supreme Court System v2")
    st.divider()
    st.markdown("**🤖 7 AI:**")
    for n in ["Gemini", "ChatGPT", "Claude", "DeepSeek", "Perplexity", "Grok", "OpenRouter"]:
        st.markdown(f'<span class="badge">• {n}</span>', unsafe_allow_html=True)
    st.divider()
    rounds = st.slider("Debate Rounds", 1, 5, 3, 
                       help="3 = सबसे बेहतर | 4-5 = गहरे सवालों के लिए")
    st.divider()
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

st.markdown('<div class="big-title">🧠 Master AI — Supreme Court</div>', unsafe_allow_html=True)
st.caption("7 AI · 3 Round Debate · Cross-Verification · Strict Judge")

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
