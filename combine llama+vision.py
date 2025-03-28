import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from transformers import GPT2TokenizerFast
from pathlib import Path
import os
import streamlit as st

# Load environment variables from .env file
load_dotenv()

# Configure Vision AI API Key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Constants for RAG
DB_FAISS_PATH = 'vectorstore/db_faiss'
MAX_CONTEXT_TOKENS = 2048
MAX_NEW_TOKENS = 512

# Tokenizer for counting tokens
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Function to initialize Vision AI model
def initialize_vision_model():
    generation_config = {"temperature": 0.3}
    return genai.GenerativeModel("gemini-1.5-flash", generation_config=generation_config)

# RAG Model Integration Functions
def set_custom_prompt():
    custom_prompt_template = """Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}

    Only return the helpful answer below and nothing else.
    Helpful answer:
    """
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt

def count_tokens(text):
    tokens = tokenizer.encode(text, return_tensors="pt")
    return tokens.shape[1]

def truncate_context(context, max_tokens):
    context_tokens = count_tokens(context)
    if context_tokens > max_tokens:
        truncated_context = tokenizer.decode(tokenizer.encode(context)[:max_tokens])
        return truncated_context
    return context

# Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

# Loading RAG Model
def load_rag_llm():
    llm = CTransformers(
        model=r"C:\Users\rathn\OneDrive\Documents\ssbot\llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.5
    )
    return llm

# QA Model Function with RAG
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_rag_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

# Function to generate content based on an image and a prompt using Vision AI
def generate_vision_content(model, image_path, prompt_text):
    image_part = {
        "mime_type": "image/jpeg",
        "data": image_path.read_bytes()
    }
    
    predefined_prompt = (
        "You are an AI specialized in solving Signals and Systems problems. "
        "The following prompt contains a question related to signals and systems. "
        "Provide an accurate and step-by-step mathematical solution for the problem. "
        "Ensure each step is clear and logically follows from the previous step."
    )
    
    combined_prompt = f"{predefined_prompt}\n\nProblem:\n{prompt_text}"
    prompt_parts = [combined_prompt, image_part]
    response = model.generate_content(prompt_parts)

    if response.candidates:
        candidate = response.candidates[0]
        if candidate.content and candidate.content.parts:
            text_part = candidate.content.parts[0]
            if text_part.text:
                return text_part.text
    return "No valid content generated."

# Output function for QA with RAG
# Output function for QA with RAG
def final_result(query):
    # Initialize the QA bot, which includes loading the retriever and model
    qa_chain = qa_bot()
    
    if qa_chain is None:
        return {"result": "FAISS index not available. Please ensure the FAISS index is correctly created."}

    # Pass the user query directly to the QA chain
    response = qa_chain({'query': query})
    return response

# Streamlit App with History Feature in Separate Tabs
def main():
    # Initialize Vision AI model
    vision_model = initialize_vision_model()

    # Initialize session state for histories if not already done
    if "vision_history" not in st.session_state:
        st.session_state.vision_history = []
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []

    st.title("Combined AI System: Vision AI and RAG QA")

    # Creating Tabs
    tab1, tab2, tab3 = st.tabs(["Vision AI", "QA with RAG", "Interaction History"])

    # Vision AI Tab
    with tab1:
        st.header("Vision AI: Image Interpretation")

        # Upload Image File
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
        prompt_text = st.text_input("Enter your prompt related to the image:")

        if uploaded_file is not None and prompt_text:
            # Process the image and prompt
            image_path = Path("temp_image.jpg")
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            vision_response = generate_vision_content(vision_model, image_path, prompt_text)

            # Display the response
            st.write("Vision AI Response:")
            st.write(vision_response)

            # Save to history
            st.session_state.vision_history.append({"prompt": prompt_text, "response": vision_response})

            # Clean up temporary file
            image_path.unlink()

    # QA with RAG Tab
    with tab2:
        st.header("QA with RAG")

        # Input Query
        query = st.text_input("Enter your question for the QA bot:")

        if query:
            response = final_result(query)

            # Display the response
            st.write("QA Bot Response:")
            st.write(response['result'])

            # Save to history
            st.session_state.qa_history.append({"query": query, "response": response['result']})

    # Interaction History Tab
    with tab3:
        st.header("Interaction History")

        # Display Vision AI History
        st.subheader("Vision AI Interaction History")
        if st.session_state.vision_history:
            for idx, interaction in enumerate(st.session_state.vision_history):
                st.write(f"**Prompt {idx + 1}:** {interaction['prompt']}")
                st.write(f"**Response {idx + 1}:** {interaction['response']}")
        else:
            st.write("No Vision AI interactions yet.")

        # Display QA with RAG History
        st.subheader("QA with RAG Interaction History")
        if st.session_state.qa_history:
            for idx, interaction in enumerate(st.session_state.qa_history):
                st.write(f"**Query {idx + 1}:** {interaction['query']}")
                st.write(f"**Response {idx + 1}:** {interaction['response']}")
        else:
            st.write("No QA with RAG interactions yet.")

if __name__ == "__main__":
    main()
