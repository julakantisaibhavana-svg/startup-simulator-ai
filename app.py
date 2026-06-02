import streamlit as st
from groq import Groq
import json
import pandas as pd
import matplotlib.pyplot as plt

# 🔥 ADD THIS (MONGODB / DB IMPORT)
from db import save_message, load_history, init_db

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="Startup Simulator AI",
    page_icon="🚀",
    layout="wide"
)

# 🔥 INIT DB (IMPORTANT)
init_db()

# ======================
# LOAD CSS
# ======================
def load_css():
    with open("styles/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

try:
    load_css()
except:
    pass

# ======================
# SESSION STATE
# ======================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ======================
# MODEL
# ======================
client = Groq(api_key=st.secrets["GROQ_API_KEY"])
model_name = "llama-3.3-70b-versatile"

# ======================
# SIDEBAR & REAL-TIME APPROACH
# ======================
st.sidebar.title("🚀 Startup Simulator")

temperature = st.sidebar.slider("Creativity", 0.0, 1.0, 0.5)

if st.sidebar.button("🗑 Reset"):
    st.session_state.messages = []
    st.rerun()

st.sidebar.markdown("---")

# 🔥 NEW SECTION: REAL-TIME STEP-BY-STEP INTERACTION APPROACH
with st.sidebar.expander("🛠️ APPROACH: Real-Time Implementation Guide", expanded=False):
    st.markdown("""
    ### **Real-Time System Architecture**
    This application executes a **4-Step Real-Time Lifecycle** for every user submission:
    """)
    
    st.image(
        "https://images.unsplash.com/photo-1531403009284-440f080d1e12?auto=format&fit=crop&w=600&q=80",
        caption="Real-Time Data Flow Pipeline",
        use_container_width=True
    )
    
    st.markdown("""
    1. **Step 1: Input & Ingestion (UI)**
       The user inputs a text-based business pitch into `st.chat_input`. The UI instantly locks down with `st.spinner` to indicate processing.
    
    2. **Step 2: Database Persistence**
       Before hitting the AI, `save_message()` writes the raw user query to MongoDB with a UTC timestamp to preserve history.
    
    3. **Step 3: Orchestrated Inference**
       The application packages the strict structural `SYSTEM_PROMPT` along with the user's concept and transmits it via local sockets to the `ollama.chat` engine running `llama3.2`.
    
    4. **Step 4: JSON Parsing & Interface Rendering**
       The raw string response is intercepted, structural validation is verified via `json.loads()`, and dynamic structural metrics, custom data tables, and interactive Matplotlib charts are rendered live.
    """)

# ======================
# HERO
# ======================
st.markdown("""
<div class="hero">
    <h1>🚀 AI Startup Investor Simulator</h1>
    <p>Get instant VC-style analysis of your startup idea</p>
</div>
""", unsafe_allow_html=True)

# ======================
# SYSTEM PROMPT
# ======================
SYSTEM_PROMPT = {
    "role": "system",
    "content": """
You are a senior Venture Capital investor. Analyze the startup idea deeply and break down your answer.

CRITICAL INSTRUCTIONS:
1. "summary" MUST be a detailed structural overview and explanation of at least 500 characters.
2. Provide a realistic total investment amount in USD if the verdict is INVEST or WATCH (0 if REJECT).
3. Provide a clear categorical breakdown of where that investment should be allocated (e.g., Product Development, Marketing, Hiring, Operations).

Return ONLY valid JSON in this exact format:
{
  "summary": "Detailed analysis breakdown... (Must be 500+ characters long)",
  "problem": ["", ""],
  "market": ["", ""],
  "business_model": ["", ""],
  "risks": ["", "", ""],
  "verdict": "INVEST | WATCH | REJECT",
  "total_investment_usd": 500000,
  "investment_allocation": {
     "Product Development": "40%",
     "Marketing & Sales": "30%",
     "Team Expansion": "20%",
     "Operations": "10%"
  },
  "scores": {
    "idea": 0,
    "market": 0,
    "execution": 0
  }
}

No extra text. Only JSON.
"""
}

# ======================
# SHOW HISTORY
# ======================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ======================
# INPUT
# ======================
prompt = st.chat_input("Enter your startup idea...")

if prompt:

    st.session_state.messages.append({"role": "user", "content": prompt})

    # 🔥 SAVE USER MESSAGE TO MONGODB/DB
    save_message("user", prompt)

    with st.chat_message("user"):
        st.markdown(prompt)

   response = client.chat.completions.create(
    model=model_name,
    messages=[
        SYSTEM_PROMPT,
        {
            "role": "user",
            "content": prompt
        }
    ],
    temperature=temperature,
    max_tokens=1200,
)

raw_output = response.choices[0].message.content
            # ======================
            # PARSE JSON
            # ======================
            try:
                data = json.loads(raw_output)
            except:
                st.error("Model did not return valid JSON. Try again.")
                st.stop()

            # ======================
            # DASHBOARD UI
            # ======================
# ======================
            # DASHBOARD UI (FIXED GRAPH SIZE)
            # ======================

            st.subheader("📊 Startup Scores")

            col1, col2, col3 = st.columns(3)
            col1.metric("Idea", f"{data['scores']['idea']}/10")
            col2.metric("Market", f"{data['scores']['market']}/10")
            col3.metric("Execution", f"{data['scores']['execution']}/10")

            df = pd.DataFrame({
                "Category": ["Idea", "Market", "Execution"],
                "Score": [
                    data["scores"]["idea"],
                    data["scores"]["market"],
                    data["scores"]["execution"]
                ]
            })

            # 🔥 CHANGE 1: Set a small, elegant dimension for the graph box
            fig, ax = plt.subplots(figsize=(5, 2.2))
            
            # 🔥 CHANGE 2: Make the bars narrower (width=0.4) so they don't look bulky
            bars = ax.bar(df["Category"], df["Score"], color="#1f77b4", width=0.4)
            
            # Formatting to make it clean
            ax.set_ylim(0, 10)
            ax.tick_params(axis='both', labelsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#cccccc')
            ax.spines['bottom'].set_color('#cccccc')
            
            # Add value labels on top of each bar
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, yval + 0.3, f"{yval}", ha='center', va='bottom', fontsize=8, fontweight='bold')

            plt.tight_layout()
            
            # 🔥 CHANGE 3: Set use_container_width=False so it stays compact on screen
            st.pyplot(fig, use_container_width=False)

            st.subheader("📢 Verdict")

            if data["verdict"] == "INVEST":
                st.success("💰 INVEST")
            elif data["verdict"] == "WATCH":
                st.warning("👀 WATCH")
            else:
                st.error("❌ REJECT")
            # INVESTMENT & CATEGORIES BREAKDOWN
            st.subheader("💸 Funding & Investment Strategy")
            if data["verdict"] in ["INVEST", "WATCH"] and data.get("total_investment_usd", 0) > 0:
                st.metric(label="Recommended Funding Amount", value=f"${data['total_investment_usd']:,} USD")
                
                st.write("**Investment Allocation Breakdown:**")
                alloc_data = {"Category": [], "Fund Share": []}
                for category, percentage in data.get("investment_allocation", {}).items():
                    alloc_data["Category"].append(category)
                    alloc_data["Fund Share"].append(percentage)
                
                alloc_df = pd.DataFrame(alloc_data)
                st.table(alloc_df)
            else:
                st.info("No funds allocated for a REJECTED verdict status.")

            # Detailed breakdowns
            st.subheader("🔍 Core Breakdown Analysis")
            tab1, tab2, tab3 = st.tabs(["Target Problem", "Market Viability", "Business Model"])
            with tab1:
                for p in data.get("problem", []):
                    st.write("•", p)
            with tab2:
                for m in data.get("market", []):
                    st.write("•", m)
            with tab3:
                for b in data.get("business_model", []):
                    st.write("•", b)

            st.subheader("🧠 Final Comprehensive Summary")
            st.info(data["summary"])
            st.caption(f"Summary length: {len(data['summary'])} characters")

            st.subheader("⚠️ Risks")
            for r in data["risks"]:
                st.write("•", r)

            # ======================
            # STORE ASSISTANT RESPONSE
            # ======================
            st.session_state.messages.append(
                {"role": "assistant", "content": raw_output}
            )

            # 🔥 SAVE AI RESPONSE TO MONGODB/DB
            save_message("assistant", raw_output)
