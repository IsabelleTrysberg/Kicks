import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Kicks Skin Guide", page_icon="✨")

@st.cache_resource
def get_vectorstore():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=st.secrets["OPENAI_API_KEY"]
    )

    return Chroma(
        persist_directory="./db_skincare_v3",
        embedding_function=embeddings
    )

@st.cache_resource
def get_llm():
    return ChatOpenAI(
        model="gpt-4o",
        api_key=st.secrets["OPENAI_API_KEY"],
        temperature=0.7
    )

def get_general_response(user_input: str) -> str:
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Du är en varm och kunnig hudvårdsbästis från Kicks. "
            "Din ton är peppig, personlig och hjälpsam. "
            "Om användaren behöver hjälp att förstå hudtyp, "
            "förklara torr, fet och kombinerad hud pedagogiskt och vänligt. "
            "Svara på svenska."
        ),
        ("human", "{input}"),
    ])

    messages = prompt.invoke({"input": user_input})
    return llm.invoke(messages).content

def get_rag_response(user_input: str, selected_skin: str | None = None) -> str:
    llm = get_llm()
    vectorstore = get_vectorstore()

    search_kwargs = {"k": 5}
    if selected_skin:
        search_kwargs["filter"] = {"skin_type": selected_skin}

    docs = vectorstore.similarity_search(user_input, **search_kwargs)
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Du är en hudvårdsexpert på Kicks. "
            "Använd endast informationen i kontexten för att svara på frågan. "
            "VIKTIGT: Om samma produkt förekommer flera gånger, presentera den bara en gång. "
            "Om pris och länk finns i kontexten ska de inkluderas. "
            "Om information saknas i kontexten ska du säga det tydligt och inte hitta på. "
            "Svara på svenska.\n\n"
            "Kontext:\n{context}"
        ),
        ("human", "{input}"),
    ])

    messages = prompt.invoke({
        "context": context if context else "Ingen relevant kontext hittades.",
        "input": user_input,
    })

    return llm.invoke(messages).content

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Men hej bästis! ✨ Vad kul att du hör av dig. Berätta, hur mår huden idag eller är det något speciellt du funderar på?"
        }
    ]

if "step" not in st.session_state:
    st.session_state.step = "chat"

if "selected_skin" not in st.session_state:
    st.session_state.selected_skin = None

st.title("✨ Din Hudvårdsbästis")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Skriv till din bästis här..."):
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        if st.session_state.selected_skin:
            response_text = get_rag_response(
                user_input=user_input,
                selected_skin=st.session_state.selected_skin
            )
        else:
            response_text = get_general_response(user_input)

        st.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})

        if "hudtyp" in response_text.lower() or "vilken typ" in response_text.lower():
            st.session_state.step = "show_skin_selection"

if st.session_state.step == "show_skin_selection" and not st.session_state.selected_skin:
    with st.expander("Klicka här för att välja din hudtyp ✨", expanded=True):
        choice = st.radio(
            "Min hudtyp är:",
            ["torr-hud", "fet-hud", "kombinerad-hud"],
            index=None
        )

        if st.button("Bekräfta hudtyp"):
            if choice:
                st.session_state.selected_skin = choice
                st.success(f"Toppen! Nu vet jag att du har {choice}. Vad vill du ha för produkttips?")
                st.rerun()
            else:
                st.warning("Välj en hudtyp först.")