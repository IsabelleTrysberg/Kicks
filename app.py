import streamlit as st
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 1. KONFIGURATION & KOPPLING TILL DIN BEFINTLIGA DB ---
st.set_page_config(page_title="Kicks Skin Guide", page_icon="✨")

# Vi använder samma embeddings-inställningar som i din notebook
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", 
    api_key=st.secrets["OPENAI_API_KEY"]
)

# Vi kopplar upp oss mot din befintliga mapp som du skapade i notebooken
vectorstore = Chroma(
    persist_directory="./db_skincare_v3", 
    embedding_function=embeddings
)

# --- 2. SESSION STATE (För att hålla koll på bästis-flödet) ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Men hej bästis! ✨ Vad kul att du hör av dig. Berätta, hur mår huden idag eller är det något speciellt du funderar på?"}]
if "step" not in st.session_state:
    st.session_state.step = "chat"
if "selected_skin" not in st.session_state:
    st.session_state.selected_skin = None

# --- 3. UI - CHATTEN ---
st.title("✨ Din Hudvårdsbästis")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. LOGIK FÖR SVAR ---
if user_input := st.chat_input("Skriv till din bästis här..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        # Vi använder GPT-4o som i din notebook men med en varmare prompt
        llm = ChatOpenAI(model="gpt-4o", api_key=st.secrets["OPENAI_API_KEY"], temperature=0.7)
        
        # Denna prompt används för det naturliga samtalet (steg 1-4)
        system_prompt_warm = (
            "Du är en varm och kunnig hudvårdsbästis från Kicks. Din ton är peppig och personlig. "
            "Använd din kunskap om hudvård för att svara på frågor. Om användaren behöver veta sin hudtyp, "
            "förklara de olika typerna (torr, fet, kombinerad) pedagogiskt och vänligt."
        )
        
        # Om vi inte har valt hudtyp än, kör vi bara vanlig chatt
        if not st.session_state.selected_skin:
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt_warm),
                ("human", "{input}"),
            ])
            chain = prompt | llm
            response_text = chain.invoke({"input": user_input}).content
        
        # Om hudtyp ÄR vald, använder vi din RAG-kedja från notebooken
        else:
            # Här skapar vi din specifika system prompt från notebooken
            rag_system_prompt = (
                "Du är en hudvårdsexpert på Kicks. Använd kontexten nedan för att svara. "
                "VIKTIGT: Presentera varje produkt bara EN gång. Inkludera pris och länk. \n\n {context}"
            )
            
            rag_prompt = ChatPromptTemplate.from_messages([
                ("system", rag_system_prompt),
                ("human", "{input}"),
            ])

            # Vi skapar en retriever med dina filter
            # För enkelhetens skull i detta steg söker vi brett på vald hudtyp
            retriever = vectorstore.as_retriever(
                search_kwargs={
                    "k": 5, 
                    "filter": {"skin_type": st.session_state.selected_skin}
                }
            )
            
            qa_chain = create_stuff_documents_chain(llm, rag_prompt)
            rag_chain = create_retrieval_chain(retriever, qa_chain)
            
            result = rag_chain.invoke({"input": user_input})
            response_text = result["answer"]

        st.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})

        # Trigga val av hudtyp om boten pratar om det
        if "hudtyp" in response_text.lower() or "vilken typ" in response_text.lower():
            st.session_state.step = "show_skin_selection"

# --- 5. DYNAMISKA VAL ---
if st.session_state.step == "show_skin_selection" and not st.session_state.selected_skin:
    with st.expander("Klicka här för att bocka i din hudtyp ✨", expanded=True):
        choice = st.radio("Min hudtyp är:", ["torr-hud", "fet-hud", "kombinerad-hud"], index=None)
        if st.button("Bekräfta"):
            st.session_state.selected_skin = choice
            st.success(f"Toppen! Nu har jag koll på att du har {choice}. Vad vill du ha för produkttips?")
            st.rerun()