import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import cuda, bfloat16
import transformers
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory

# Constants
FDA_DOCUMENTS_PATH = "/content/FDA_documents"
VECTOR_DB_PATH = "/content/Vector_db_dir"

# Load and process documents
@st.cache_resource
def load_and_process_documents():
    try:
        loader = DirectoryLoader(
            path=FDA_DOCUMENTS_PATH,
            glob="*.txt",
            loader_cls=UnstructuredFileLoader
        )
        documents = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
        text_chunks = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings()

        vectordb = Chroma.from_documents(
            documents=text_chunks,
            embedding=embeddings,
            persist_directory=VECTOR_DB_PATH
        )

        return vectordb
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")
        return None

# Initialize LLaMA or GPT-J model
@st.cache_resource
def initialize_model():
    # Quantization configuration using bitsandbytes
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )

    try:
        model_id = "meta-llama/Llama-2-7b-chat-hf"  # You can change to GPT-J if needed
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error initializing LLaMA model: {str(e)}")
        return None, None

# # Generate answer using the LLM
# def generate_answer(query, retrieved_docs, model, tokenizer):
#     context = " ".join(retrieved_docs)  # Combine retrieved docs
#     prompt = f"Answer the following Question: {query}\n Given the following extracted \nContext: {context}\n\nAnswer:"

#     inputs = tokenizer(prompt, return_tensors="pt")
#     outputs = model.generate(**inputs, max_new_tokens=200)

#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Generate answer using the LLM
def generate_answer(query, retrieved_docs, model, tokenizer):
    context = " ".join(retrieved_docs)  # Combine retrieved docs into context
    prompt = f"Answer the question based on the context:\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"

    # Tokenize the prompt and convert to tensor
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate only the answer
    outputs = model.generate(**inputs, max_new_tokens=200)

    # Decode and return only the generated answer
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Post-process to extract just the part after "Answer:" if needed
    answer = answer.split("Answer:")[-1].strip()

    return answer

# Streamlit app
def main():
    st.title("PANDAbot")

    vectordb = load_and_process_documents()
    if vectordb is None:
        st.error("Failed to load documents. Please check your document path and try again.")
        return

    model, tokenizer = initialize_model()
    if model is None or tokenizer is None:
        st.error("Failed to initialize model. Please check your setup and try again.")
        return

    # Chat interface
    st.sidebar.header("Chat History")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # if prompt := st.chat_input("Ask a question about FDA documents"):
    #     st.session_state.messages.append({"role": "user", "content": prompt})
    #     with st.chat_message("user"):
    #         st.markdown(prompt)

    #     with st.chat_message("assistant"):
    #         try:
    #             retriever = vectordb.as_retriever()
    #             retrieved_docs = [doc.page_content for doc in retriever.get_relevant_documents(prompt)]
    #             response = generate_answer(prompt, retrieved_docs, model, tokenizer)
    #             st.markdown(response)
    #             st.session_state.messages.append({"role": "assistant", "content": response})
    #         except Exception as e:
    #             error_message = f"An error occurred: {str(e)}"
    #             st.error(error_message)
    #             st.session_state.messages.append({"role": "assistant", "content": error_message})

    # st.sidebar.markdown("\n".join([f"**{m['role']}**: {m['content']}" for m in st.session_state.messages]))

    if prompt := st.chat_input("Ask a question about FDA documents"):
      st.session_state.messages.append({"role": "user", "content": prompt})
      with st.chat_message("user"):
          st.markdown(prompt)

      with st.chat_message("assistant"):
          try:
              retriever = vectordb.as_retriever()
              retrieved_docs = [doc.page_content for doc in retriever.get_relevant_documents(prompt)]
              response = generate_answer(prompt, retrieved_docs, model, tokenizer)
              st.markdown(response)  # Display only the final answer
              st.session_state.messages.append({"role": "assistant", "content": response})
          except Exception as e:
              error_message = f"An error occurred: {str(e)}"
              st.error(error_message)
              st.session_state.messages.append({"role": "assistant", "content": error_message})
                
      st.sidebar.markdown("\n".join([f"**{m['role']}**: {m['content']}" for m in st.session_state.messages]))

if __name__ == "__main__":
    main()
