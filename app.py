import random
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Hudvårdstips från Kicks", page_icon="✨")
st.markdown("Din hudvårdsbästis som hjälper dig hitta rätt produkter för just din hud 💖")

GREETINGS = [
    "Hej gorgeous! ✨ Jag hjälper dig hitta en rutin som passar just din hud!",
    "Hej där! Jag hjälper dig gärna med din hudvård. Tell me all about it!",
    "What's cooking, good looking? 💖 Jag är här för att hjälpa dig med din hudvård!",
    "Hej snygging! ✨ Berätta hur din hud mår så fixar vi en rutin som passar dig perfekt!",
]

SKIN_TYPES = ["torr-hud", "fet-hud", "kombinerad-hud"]
PRODUCT_CATEGORIES = ["rengöring", "serum", "ansiktskräm"]

BASE_SYSTEM_PROMPT = (
    "Du är en varm, charmig och vänskaplig hudvårdsbästis från Kicks. "
    "Svara på svenska och håll en personlig, peppig och lite flirtig ton.\n\n"
    "VIKTIGT:\n"
    "- Du är mitt i en pågående konversation.\n"
    "- Hälsa inte igen.\n"
    "- Svara naturligt som en fortsättning i en konversation.\n"
)


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
        "faiss_skincare_v2",
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


def get_general_response(user_input: str) -> str:
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            BASE_SYSTEM_PROMPT +
            "Småprata gärna naturligt, men ge inte produkttips om användaren inte ber om hjälp med hudvård."
        ),
        ("human", "{input}"),
    ])

    messages = prompt.invoke({"input": user_input})
    return llm.invoke(messages).content


def get_skin_type_response(user_input: str) -> str:
    llm = get_llm()
    vectorstore = get_vectorstore()

    candidate_docs = vectorstore.similarity_search(user_input, k=20)

    docs = [
        doc for doc in candidate_docs
        if doc.metadata.get("source") == "olika-hudtyper-masterguide.txt"
    ][:5]

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            BASE_SYSTEM_PROMPT +
            "Du får ENDAST använda information från kontexten.\n"
            "- Du får inte gissa eller hitta på något.\n"
            "- Du får inte ge produkttips i detta steg.\n"
            "- Avsluta alltid med att be användaren bekräfta sin hudtyp i rutorna nedanför.\n\n"
            "Exempel på ton:\n"
            "'Jag förstår! Det du beskriver låter som...'\n"
            "'Det kan tyda på...'\n"
            "Kontext:\n{context}"
        ),
        ("human", "{input}"),
    ])

    messages = prompt.invoke({
        "context": context if context else "Ingen relevant kontext hittades.",
        "input": user_input,
    })

    return llm.invoke(messages).content


def build_skin_type_helper_response() -> str:
    return (
        "För att ge dig mina bästa tips behöver du först bekräfta din hudtyp nedan 💖\n\n"
        "**Torr hud** – känns ofta stram, kan flaga och vill ha mycket fukt.\n\n"
        "**Fet hud** – blir lätt glansig, särskilt i T-zonen, och kan få tilltäppta porer.\n\n"
        "**Kombinerad hud** – både och, ofta fetare i panna/näsa/haka men torrare på kinderna.\n\n"
        "Vad känner du mest igen dig i? ✨"
    )


def get_post_skin_selection_response(selected_skin: str) -> str:
    llm = get_llm()
    vectorstore = get_vectorstore()

    query_map = {
        "torr-hud": "torr hud saknar fukt och fett återfuktande produkter rengöring ansiktskräm serum",
        "fet-hud": "fet hud överproduktion av talg hålla huden ren porer rengöring ansiktskräm serum",
        "kombinerad-hud": "kombinerad hud fet t-zon torrare kinder produkter utformade för kombinerad hud rengöring ansiktskräm serum",
    }

    candidate_docs = vectorstore.similarity_search(query_map[selected_skin], k=20)

    docs = [
        doc for doc in candidate_docs
        if doc.metadata.get("source") == "olika-hudtyper-masterguide.txt"
    ][:5]

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            BASE_SYSTEM_PROMPT +
            "Du får ENDAST använda information från kontexten.\n"
            "- Bekräfta den valda hudtypen på ett mjukt och naturligt sätt.\n"
            "- Beskriv kort vad som kännetecknar hudtypen.\n"
            "- Säg att en bra grund är rengöring, ansiktskräm och gärna serum.\n"
            "- Fråga användaren vad hen vill ha tips om först, men nämn att hen kan välja flera.\n"
            "- Ge INTE några konkreta produkttips ännu.\n"
            "Kontext:\n{context}"
        ),
        (
            "human",
            "Användaren har valt hudtypen: {selected_skin}"
        ),
    ])

    messages = prompt.invoke({
        "context": context if context else "Ingen relevant kontext hittades.",
        "selected_skin": selected_skin,
    })

    return llm.invoke(messages).content


def get_rag_response(user_input: str, selected_skin: str | None, selected_category: str | None = None) -> str:
    llm = get_llm()
    vectorstore = get_vectorstore()

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
            BASE_SYSTEM_PROMPT +
            "Du måste ENDAST använda information som finns i kontexten. "
            "Du får inte hitta på produkter, kategorier, priser, länkar eller råd som inte stöds av kontexten. "
            "Om exakt rätt kategori inte finns i kontexten men det finns andra relevanta produkter för samma hudtyp, "
            "ska du säga det tydligt, till exempel: "
            "'Jag hittade tyvärr ingen ansiktskräm för fet hud i mina produkter, men här är ett serum för fet hud.' "
            "Om ingen relevant kontext alls finns ska du säga det tydligt och be användaren förtydliga sitt behov. "
            "Presentera inte samma produkt flera gånger. "
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


def get_product_recommendations(selected_skin: str, selected_categories: list[str]) -> str:
    llm = get_llm()
    vectorstore = get_vectorstore()

    all_docs = []

    for category in selected_categories:
        query = f"{category} {selected_skin} produkttips"

        candidate_docs = vectorstore.similarity_search(query, k=25)

        filtered_docs = []
        for doc in candidate_docs:
            source = doc.metadata.get("source", "")
            skin_ok = doc.metadata.get("skin_type") == selected_skin
            category_ok = doc.metadata.get("category") == category
            is_master = source == "olika-hudtyper-masterguide.txt"

            if skin_ok and category_ok and not is_master:
                filtered_docs.append(doc)

        all_docs.extend(filtered_docs[:5])

    unique_docs = []
    seen = set()
    for doc in all_docs:
        key = (
            doc.metadata.get("source", ""),
            doc.metadata.get("skin_type", ""),
            doc.metadata.get("category", ""),
            doc.page_content[:200]
        )
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    context = "\n\n".join(doc.page_content for doc in unique_docs[:10])
    categories_text = ", ".join(selected_categories)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            BASE_SYSTEM_PROMPT +
            "Du får ENDAST använda information från kontexten.\n"
            "- Du får inte hitta på produkter, priser, länkar eller egenskaper.\n"
            "- Du ska bara ge tips från produktdokumenten i kontexten.\n"
            "- Om det finns länk i kontexten ska du ta med länken i svaret.\n"
            "- Om det finns pris i kontexten ska du ta med priset i svaret.\n"
            "- Om användaren valt flera kategorier får du gärna dela upp svaret per kategori.\n"
            "- Presentera inte samma produkt flera gånger.\n"
            "- Om någon vald kategori saknar produkter i kontexten ska du säga det tydligt men vänligt.\n"
            "Användaren har hudtyp: {selected_skin}\n"
            "Användaren vill ha tips om: {categories_text}\n\n"
            "Kontext:\n{context}"
        ),
        (
            "human",
            "Ge mig produkttips för de valda kategorierna."
        ),
    ])

    messages = prompt.invoke({
        "context": context if context else "Ingen relevant kontext hittades.",
        "selected_skin": selected_skin,
        "categories_text": categories_text,
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

if "need_product_selection" not in st.session_state:
    st.session_state.need_product_selection = False

if "selected_categories" not in st.session_state:
    st.session_state.selected_categories = []


st.title("✨ Din Hudvårdsbästis")

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="✨" if message["role"] == "assistant" else "💬"):
        st.markdown(message["content"])


if user_input := st.chat_input("Skriv till din bästis här..."):
    add_message("user", user_input)

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Letar fram något fabulous till dig... ✨"):

            if st.session_state.selected_skin and not st.session_state.need_product_selection:
                st.session_state.last_requested_category = infer_requested_category(user_input)
                response_text = get_rag_response(
                    user_input=user_input,
                    selected_skin=st.session_state.selected_skin,
                    selected_category=st.session_state.last_requested_category,
                )
            else:
                if user_is_asking_for_skin_help(user_input):
                    response_text = get_skin_type_response(user_input)
                    st.session_state.need_skin_selection = True
                    st.session_state.last_requested_category = infer_requested_category(user_input)
                else:
                    response_text = get_general_response(user_input)

        st.markdown(response_text)
        add_message("assistant", response_text)


if st.session_state.need_skin_selection and not st.session_state.selected_skin:
    with st.expander("Välj din hudtyp ✨", expanded=True):
        st.markdown(build_skin_type_helper_response())

        choice = st.radio(
            "Det här känner jag mest igen mig i:",
            SKIN_TYPES,
            index=None
        )

        if st.button("Bekräfta hudtyp"):
            if choice:
                st.session_state.selected_skin = choice
                st.session_state.need_skin_selection = False
                st.session_state.need_product_selection = True

                followup = get_post_skin_selection_response(choice)
                add_message("assistant", followup)
                st.rerun()
            else:
                st.warning("Välj en hudtyp först.")


if st.session_state.selected_skin:
    if st.button("🔄 Tillbaka till val av hudtyp"):
        st.session_state.selected_skin = None
        st.session_state.need_skin_selection = True
        st.session_state.need_product_selection = False
        st.session_state.selected_categories = []

        add_message(
            "assistant",
            "Okej babe, vi börjar om ✨ Välj din hudtyp igen så hittar vi rätt vibe för din hud 💖"
        )
        st.rerun()


if st.session_state.need_product_selection and st.session_state.selected_skin:
    with st.expander("Välj vad du vill ha tips om ✨", expanded=True):
        selected = []
        col1, col2, col3 = st.columns(3)

        with col1:
            rengoring = st.checkbox(
                "Rengöring",
                value="rengöring" in st.session_state.selected_categories
            )

        with col2:
            serum = st.checkbox(
                "Serum",
                value="serum" in st.session_state.selected_categories
            )

        with col3:
            ansiktskram = st.checkbox(
                "Ansiktskräm",
                value="ansiktskräm" in st.session_state.selected_categories
            )

        if rengoring:
            selected.append("rengöring")
        if serum:
            selected.append("serum")
        if ansiktskram:
            selected.append("ansiktskräm")

        if st.button("Visa produkttips"):
            if selected:
                st.session_state.selected_categories = selected
                st.session_state.need_product_selection = False

                response_text = get_product_recommendations(
                    selected_skin=st.session_state.selected_skin,
                    selected_categories=selected
                )

                add_message("assistant", response_text)
                st.rerun()
            else:
                st.warning("Välj minst en kategori först.")