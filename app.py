import random
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Hudvårdstips från Kicks", page_icon="✨")

GREETINGS = [
    "Hej gorgeous! ✨ Vad kan jag hjälpa dig med idag?",
    "Hej där! Let's make you fabulous ✨ Vad vill du ha hjälp med?",
    "What's cooking, good looking? 💖 Berätta vad huden behöver idag!",
    "Hej snygging! ✨ Vad har du på hjärtat?",
]

SKIN_TYPES = ["torr-hud", "fet-hud", "kombinerad-hud"]

@st.cache_resource
def get_embeddings():
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=st.secrets["OPENAI_API_KEY"]
    )

@st.cache_resource
def get_vectorstore():
    embeddings = get_embeddings()
    return FAISS.load_local(
        "faiss_skincare_v1",
        embeddings,
        allow_dangerous_deserialization=True
    )

@st.cache_resource
def get_llm():
    return ChatOpenAI(
        model="gpt-4o",
        api_key=st.secrets["OPENAI_API_KEY"],
        temperature=0.5
    )

def add_message(role: str, content: str):
    st.session_state.messages.append({"role": role, "content": content})

def infer_requested_category(user_input: str):
    text = user_input.lower()
    if "serum" in text:
        return "serum"
    if "rengör" in text or "tvätt" in text:
        return "rengöring"
    if "ansiktskräm" in text or "kräm" in text or "fuktkräm" in text:
        return "ansiktskräm"
    return None

def user_is_asking_for_skin_help(user_input: str) -> bool:
    text = user_input.lower()
    triggers = [
        "hud", "hudvård", "finnar", "akne", "torr", "fet", "kombinerad",
        "serum", "rengöring", "ansiktskräm", "kräm", "porer", "glansig",
        "flagar", "rodnad", "hy", "plitor"
    ]
    return any(t in text for t in triggers)

def get_skin_type_response(user_input: str) -> str:
    llm = get_llm()
    vectorstore = get_vectorstore()

    docs = vectorstore.similarity_search(
        user_input,
        k=5,
        filter={"source": "olika-hudtyper-masterguide.txt"}
    )

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Du är en varm, charmig och vänskaplig hudvårdsbästis från Kicks. "
            "Svara på svenska och håll en personlig, peppig ton.\n\n"

            "VIKTIGT:\n"
            "- Du får ENDAST använda information från kontexten.\n"
            "- Du får inte gissa eller hitta på något.\n"
            "- Du får inte ge produkttips i detta steg.\n\n"

            "Din uppgift är att:\n"
            "1. Tolka användarens beskrivning av sin hud.\n"
            "2. Koppla det till rätt hudtyp utifrån kontexten.\n"
            "3. Förklara detta pedagogiskt, t.ex:\n"
            "   'Det låter som att din hud är torr, vilket ofta kännetecknas av...'\n\n"

            "4. Hjälpa användaren att förstå sin hudtyp och bekräfta den.\n"
            "5. Ställ följdfrågor om något är oklart.\n\n"

            "Du ska INTE gå vidare till produktrekommendationer ännu.\n\n"

            "Kontext:\n{context}"
        ),
        ("human", "{input}"),
    ])

    messages = prompt.invoke({"input": user_input})
    return llm.invoke(messages).content

def build_skin_type_helper_response() -> str:
    return (
        "Jag förstår! För att ge dig de bästa tipsen vill jag först veta lite mer om din hudtyp 💖\n\n"
        "**Torr hud** – känns ofta stram, kan flaga och vill ha mycket fukt.\n\n"
        "**Fet hud** – blir lätt glansig, särskilt i T-zonen, och kan få tilltäppta porer.\n\n"
        "**Kombinerad hud** – både och, ofta fetare i panna/näsa/haka men torrare på kinderna.\n\n"
        "Välj det som känns mest som din hud här nedanför så guidar jag dig vidare ✨"
    )

def get_rag_response(user_input: str, selected_skin: str | None, selected_category: str | None = None) -> str:
    llm = get_llm()
    vectorstore = get_vectorstore()

    # Hämta brett först, filtrera sedan strikt i Python.
    candidate_docs = vectorstore.similarity_search(user_input, k=20)

    strict_docs = []
    fallback_docs = []

    for doc in candidate_docs:
        skin_ok = (selected_skin is None or doc.metadata.get("skin_type") == selected_skin)
        category_ok = (selected_category is None or doc.metadata.get("category") == selected_category)

        if skin_ok and category_ok:
            strict_docs.append(doc)

        if skin_ok:
            fallback_docs.append(doc)

    if strict_docs:
        final_docs = strict_docs[:5]
        context_mode = "strict"
    elif fallback_docs:
        final_docs = fallback_docs[:5]
        context_mode = "fallback"
    else:
        final_docs = []
        context_mode = "none"

    context = "\n\n".join(doc.page_content for doc in final_docs)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Du är en varm hudvårdsrådgivare från Kicks. "
            "Du måste ENDAST använda information som finns i kontexten. "
            "Du får inte hitta på produkter, kategorier, priser, länkar eller råd som inte stöds av kontexten. "
            "Om exakt rätt kategori inte finns i kontexten men det finns andra relevanta produkter för samma hudtyp, "
            "ska du säga det tydligt, till exempel: "
            "'Jag hittade tyvärr ingen ansiktskräm för fet hud i mina produkter, men här är ett serum för fet hud.' "
            "Om ingen relevant kontext alls finns ska du säga det tydligt och be användaren förtydliga sitt behov. "
            "Presentera inte samma produkt flera gånger. "
            "Svara på svenska, varmt och naturligt.\n\n"
            "Kontextläge: {context_mode}\n\n"
            "Kontext:\n{context}"
        ),
        ("human", "{input}"),
    ])

    messages = prompt.invoke({
        "context": context if context else "Ingen relevant kontext hittades.",
        "context_mode": context_mode,
        "input": user_input,
    })

    return llm.invoke(messages).content

# ----- STATE -----
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": random.choice(GREETINGS)}
    ]

if "selected_skin" not in st.session_state:
    st.session_state.selected_skin = None

if "need_skin_selection" not in st.session_state:
    st.session_state.need_skin_selection = False

if "last_requested_category" not in st.session_state:
    st.session_state.last_requested_category = None

st.title("✨ Din Hudvårdsbästis")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Skriv till din bästis här..."):
    add_message("user", user_input)

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        if st.session_state.selected_skin:
            st.session_state.last_requested_category = infer_requested_category(user_input)
            response_text = get_rag_response(
                user_input=user_input,
                selected_skin=st.session_state.selected_skin,
                selected_category=st.session_state.last_requested_category,
            )
        else:
            if user_is_asking_for_skin_help(user_input):
                response_text = build_skin_type_helper_response()
                st.session_state.need_skin_selection = True
                st.session_state.last_requested_category = infer_requested_category(user_input)
            else:
                response_text = get_general_response(user_input)

        st.markdown(response_text)
        add_message("assistant", response_text)

if st.session_state.need_skin_selection and not st.session_state.selected_skin:
    with st.expander("Välj din hudtyp ✨", expanded=True):
        choice = st.radio(
            "Det här känner jag mest igen mig i:",
            SKIN_TYPES,
            index=None
        )

        if st.button("Bekräfta hudtyp"):
            if choice:
                st.session_state.selected_skin = choice
                st.session_state.need_skin_selection = False

                followup = (
                    f"Perfekt, nu vet jag att du har **{choice}** ✨ "
                    "Berätta vad du vill ha hjälp med så håller jag mig till de produkter jag faktiskt har underlag för."
                )
                add_message("assistant", followup)
                st.rerun()
            else:
                st.warning("Välj en hudtyp först.")