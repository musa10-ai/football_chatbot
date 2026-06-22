import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
# Switched from Ollama to ChatGroq for Cloud Deployment
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA

# ---------------- CONFIG ----------------
DB_PATH = "vectordb"

st.set_page_config(
    page_title="Football Knowledge Chatbot",
    page_icon="⚽",
    layout="centered"
)

# ---------------- UI HEADER ----------------
st.markdown(
    """
    <h1 style='text-align: center;'>⚽ Football Knowledge Chatbot</h1>
    <p style='text-align: center; color: gray;'>
    Ask about football rules, history, tactics, clubs & legendary moments
    </p>
    """,
    unsafe_allow_html=True
)

# ---------------- LOAD RAG ----------------
@st.cache_resource
def load_rag():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # FIXED: Explicitly using groq_api_key to satisfy validation rules
    llm = ChatGroq(
    groq_api_key=st.secrets["GROQ_API_KEY"],
    model_name="openai/gpt-oss-20b",
    temperature=0
)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

    return qa

qa = load_rag()

# ---------------- CHAT STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- DISPLAY CHAT ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- INPUT ----------------
user_input = st.chat_input("Ask a football question... ⚽")

if user_input:
    # show user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    # generate answer
    with st.chat_message("assistant"):
        with st.spinner("Analyzing football knowledge..."):
            result = qa.invoke({"query": user_input})
            answer = result["result"]

            st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

# ---------------- FOOTER ----------------
st.markdown(
    """
    <hr>
    <p style='text-align:center; font-size:12px; color:gray;'>
    ⚽ Cloud RAG Football Bot | Powered by Groq + FAISS
    </p>
    """,
    unsafe_allow_html=True
)
