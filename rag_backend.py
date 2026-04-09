import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma


DB_PATH = "./db_skincare_v3"


@st.cache_resource
def load_vectorstore():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=st.secrets["OPENAI_API_KEY"]
    )

    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )
    return vectorstore


@st.cache_resource
def load_llm():
    return ChatOpenAI(
        model="gpt-4o",
        api_key=st.secrets["OPENAI_API_KEY"]
    )


def get_rag_response(user_question, selected_skin_type=None, selected_category=None):
    llm = load_llm()
    vectorstore = load_vectorstore()

    # Bygg filter dynamiskt utifrån användarens val
    filters = []

    if selected_skin_type and selected_skin_type != "Välj...":
        filters.append({"skin_type": selected_skin_type})

    if selected_category and selected_category != "Välj...":
        filters.append({"category": selected_category})

    search_kwargs = {"k": 5}

    if len(filters) == 1:
        search_kwargs["filter"] = filters[0]
    elif len(filters) > 1:
        search_kwargs["filter"] = {"$and": filters}

    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

    system_prompt = (
        "Du är en hudvårdsexpert på Kicks. "
        "Använd kontexten nedan för att svara på frågan. "
        "VIKTIGT: Om samma produkt förekommer flera gånger i kontexten, presentera den bara EN gång. "
        "Gör svaret koncist och pedagogiskt. "
        "Om pris och länk finns i kontexten ska de inkluderas. "
        "Om de inte finns i kontexten ska du inte hitta på dem."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": user_question})
    return response["answer"]