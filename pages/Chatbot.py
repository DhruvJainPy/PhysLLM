import streamlit as st
import time
import random
import os
import base64
from Utils.GenerateAnswer import generate_answer, classify_query, llm
from dotenv import load_dotenv

load_dotenv()

# --- Page Config ---
st.set_page_config(
    page_title="PhysLLM",
    page_icon="Utils/logo.jpg" if os.path.exists("Utils/logo.jpg") else "⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
    
)

LOGO_PATH = "Utils/logo.jpg"
has_logo = os.path.exists(LOGO_PATH)
assistant_avatar = LOGO_PATH if has_logo else "⚛️"

def get_img_as_base64(file_path):
    """Reads an image file and converts it to a base64 string."""
    if not os.path.exists(file_path):
        return ""
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

logo_base64 = get_img_as_base64(LOGO_PATH) if has_logo else ""

# --- Custom CSS ---
st.markdown("""
<style>
    header {visibility: visible !important;}

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main container styling */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 8rem; /* Padding to ensure content isn't hidden behind footer */
    }
    
    /* --- CHAT INPUT STYLING (IMPROVED) --- */
    
    /* Style the bottom container (the sticky footer) */
    div[data-testid="stBottom"] {
        background-color: transparent !important;
        border: none !important;
    }
    

    /* Style the inner chat input box (Textarea) */
    .stChatInput textarea {
        border: none; /* Subtle border */
        color: #262730 !important;
        border-radius: 25px !important; /* PILL SHAPE */
        padding: 12px 20px !important; /* Comfortable internal padding */
        transition: all 0.2s ease;
    }
    
    /* Focus state for the input box */
    .stChatInput textarea:focus {
        outline: none !important;
    }
    
    /* Adjust the send button icon color */
    .stChatInput button svg {
        color: #5D5D5D !important;
        margin-bottom: 4px;
    }

    /* Message styling */
    .stChatMessage {
        background-color: transparent;
    }
    
    .stChatMessage:hover {
        background-color: #F6F1E7;
        border-radius: 10px;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #F6F1E7;
        border-right: 1px solid #EAEAEA;
    }

    /* --- INFO CARD FIXES --- */
    div[data-testid="stAlert"] {
        background-color: #F6F1E7 !important; 
        border: 1px solid #E0D8CC !important; 
        color: #262730 !important;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    
    div[data-testid="stAlert"] > div {
        background-color: transparent !important;
        color: #262730 !important;
    }

    div[data-testid="stAlert"] svg {
        fill: #5D5D5D !important;
        color: #5D5D5D !important;
    }
    
    /* --- SIDEBAR BUTTON STYLING (NEW CHAT) --- */
    [data-testid="stSidebar"] .stButton button[kind="primary"] {
        background-color: transparent !important;
        color: #4a4a4a !important;
        width: 100%;
        height: 24;
        border: none !important;
        padding-left: 24px !important;
        padding-right: 20px !important;
        text-align: left !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important; 
        transition: all 0.2s ease;
        font-weight: 500;
        font-size: 15px;
    }
    
    [data-testid="stSidebar"] .stButton button[kind="primary"]:hover {
        color: blue !important;
    }
    
    [data-testid="stSidebar"] .stButton button[kind="primary"]:active {
        transform: none;
        background-color: #F9F3E6 !important;
    }

    /* Style the "History" buttons (Secondary) */
    [data-testid="stSidebar"] .stButton button[kind="secondary"] {
        background-color: transparent;
        border: none;
        color: #4a4a4a;
        text-align: left;
        display: block;
        width: 100%;
        padding-left: 10px;
        font-weight: normal;
    }
    
    [data-testid="stSidebar"] .stButton button[kind="secondary"]:hover {
        background-color: #EAE4D9; 
        color: #000;
        border-radius: 5px;
    }
    
    [data-testid="stSidebar"] button:focus {
        box-shadow: none !important;
        border-color: transparent !important;
    }
    
</style>
""", unsafe_allow_html = True)

# --- State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar ---
with st.sidebar:
    st.markdown("""
    <style>
        /* Sidebar safe styling */
        section[data-testid="stSidebar"] {
            background-color: #F6F1E7 !important;
            border-right: 1px solid #EAEAEA !important;
            padding-top: 1rem;
        }

        .sidebar-title {
            font-size: 0.8rem;
            font-weight: 600;
            color: #555;
            margin-top: 1rem;
            margin-bottom: 0.2rem;
            letter-spacing: 0.5px;
        }

        .sidebar-item {
            padding: 6px 8px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.85rem;
        }

        .sidebar-item:hover {
            background-color: #EAE4D9;
        }

    </style>
    """, unsafe_allow_html=True)

    if has_logo:
        st.image(LOGO_PATH, width=200)
    else:
        st.title("⚛️ PhysLLM")

    # ---- New Chat ----
    if st.button("➕  New Chat", key="new_chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

    history_items = [
    ]

    for i, item in enumerate(history_items):
        if st.button(item, key=f"hist_{i}", help="Load past conversation"):
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"(History) Loaded: {item}"
            })
            st.rerun()



    
# --- Main Content ---

# Display welcome message if chat is empty
if not st.session_state.messages:
    if has_logo:
        st.markdown(f"""
        <div style='text-align: center; margin-top: 50px; margin-bottom: 20px; display: flex; align-items: center; justify-content: center; '>
            <h1 style='margin-left: 10px;'>Physics Made Simple with PhysLLM</h1>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("⚙️ **Mechanics**\n\nAnalyze motion, forces, and energy systems.\n\n")
    with col2:
        st.info("⚡ **Electricity & Magnetism**\n\nExplore fields, circuits, and electromagnetism.\n\n")
    with col3:
        st.info("🌊 **Waves & Modern Physics**\n\nDive into optics, relativity, and quantum theory.\n\n")

# Render existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else assistant_avatar):
        st.markdown(message["content"])

import concurrent.futures

# Global thread executor for background tasks
executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

def run_rag_background(query):
    """Runs RAG (generate_answer) in a background thread."""
    try:
        decision = classify_query(query)
        if decision == "CONV_ONLY":
            result = llm.invoke(query)
            return result.content
        else:
            return generate_answer(query)
    except Exception as e:
        return f"Error: {e}"

def stream_response(prompt):
    """Runs RAG in background and streams result chunk-by-chunk."""
    user_query = prompt.strip()
    if not user_query:
        yield ""
        return

    # 🔥 Submit heavy work to background thread
    future = executor.submit(run_rag_background, user_query)

    # 🔥 Wait for result without blocking UI
    while future.running():
        time.sleep(0.05)
        yield ""  # keeps UI responsive

    result = future.result()

    # Stream result word-by-word
    for word in result.split(" "):
        yield word + " "
        time.sleep(0.01)


# --- User Input Handling ---
if prompt := st.chat_input("Ask PhysLLM a physics question..."):
    # 1. User Message
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. AI Response using st.write_stream (Smoother performance)
    with st.chat_message("assistant", avatar=assistant_avatar):
        full_response = st.write_stream(stream_response(prompt))
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})