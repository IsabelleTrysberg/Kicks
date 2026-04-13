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
PRODUCT_CATEGORIES = ["rengöring", "serum", "ansiktskräm"]


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


def infer_requested_categories(user_input: str):
    text = user_input.lower()
    categories = []

    if "serum" in text:
        categories.append("serum")

    if "rengör" in text or "tvätt" in text:
        categories.append("rengöring")

    if "ansiktskräm" in text or "kräm" in text or "fuktkräm" in text:
        categories.append("ansiktskräm")

    if "alla" in text or "allt" in text or "hela rutinen" in text:
        return PRODUCT_CATEGORIES.copy()

    # ta bort dubletter men behåll ordning
    seen = set()
    unique_categories = []
    for cat in categories:
        if cat not in seen:
            seen.add(cat)
            unique_categories.append(cat)

    return unique_categories


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
            "Du är en varm, charmig och vänskaplig hudvårdsbästis från Kicks. "
            "Svara på svenska och håll en personlig, peppig ton. "
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
            "Du är en varm, charmig och vänskaplig hudvårdsbästis från Kicks. "
            "Svara på svenska och håll en personlig, peppig ton.\n\n"

            "VIKTIGT:\n"
            "- Du får ENDAST använda information från kontexten.\n"
            "- Du får inte gissa eller hitta på något.\n"
            "- Du får inte ge produkttips i detta steg.\n"
            "- Du ska hjälpa användaren att förstå vilken hudtyp det låter som att hen har.\n"
            "- Om något är oklart ska du uttrycka dig försiktigt.\n"
            "- Avsluta alltid med att be användaren bekräfta sin hudtyp i rutorna nedanför.\n\n"

            "Exempel på ton:\n"
            "'Jag förstår! Det du beskriver låter som...'\n"
            "'Det kan tyda på...'\n"
            "'Bekräfta gärna din hudtyp nedan så plockar jag fram mina bästa tips.'\n\n"

            "Kontext:\n{context}"
        ),
        ("human", "{input}"),
    ])

    messages = prompt.invoke({
        "context": context if context else "Ingen relevant kontext hittades.",
        "input": user_input,
    })

    return llm.invoke(messages).content


def get_skin_confirmation_response(selected_skin: str) -> str:
    llm = get_llm()
    vectorstore = get_vectorstore()

    query_map = {
        "torr-hud": "torr hud vad kännetecknar torr hud och vilka produkter är viktiga",
        "fet-hud": "fet hud vad kännetecknar fet hud och vilka produkter är viktiga",
        "kombinerad-hud": "kombinerad hud vad kännetecknar kombinerad hud och vilka produkter är viktiga",
    }

    candidate_docs = vectorstore.similarity_search(query_map[selected_skin], k=10)

    docs = [
        doc for doc in candidate_docs
        if doc.metadata.get("source") == "olika-hudtyper-masterguide.txt"
    ][:5]

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Du är en varm, charmig, flirtig och vänskaplig hudvårdsbästis från Kicks. "
            "Du ska låta som en peppig stylist-kompis som ger komplimanger och får användaren att känna sig fabulous. "
            "Svara på svenska och håll en mjuk, flytande och personlig ton.\n\n"

            "VIKTIGT:\n"
            "- Du får ENDAST använda information från kontexten.\n"
            "- Bekräfta den valda hudtypen på ett varmt, naturligt och lite flirtigt sätt.\n"
            "- Beskriv kort vad som kännetecknar hudtypen.\n"
            "- Säg att rengöring, ansiktskräm och serum är en bra grund för att ta hand om huden.\n"
            "- Fråga användaren vilken produkt hen vill ha tips om: rengöring, serum, ansiktskräm eller alla.\n"
            "- Nämn inte produkter vid namn i detta steg.\n"
            "- Håll svaret flytande och trevligt, inte stelt eller punktigt.\n"
            "- Det ska kännas som ett chattmeddelande från en peppig hudvårdsbästis.\n\n"

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


def build_skin_type_helper_response() -> str:
    return (
        "Jag förstår! För att ge dig de bästa tipsen vill jag först veta lite mer om din hudtyp 💖\n\n"
        "**Torr hud** – känns ofta stram, kan flaga och vill ha mycket fukt.\n\n"
        "**Fet hud** – blir lätt glansig, särskilt i T-zonen, och kan få tilltäppta porer.\n\n"
        "**Kombinerad hud** – både och, ofta fetare i panna/näsa/haka men torrare på kinderna.\n\n"
        "Välj det som känns mest som din hud här nedanför så guidar jag dig vidare ✨"
    )


def get_rag_response(
    user_input: str,
    selected_skin: str | None,
    selected_category: str | None = None
) -> str:
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


def get_rag_response_multi(
    user_input: str,
    selected_skin: str,
    selected_categories: list[str] | None = None
) -> str:
    llm = get_llm()
    vectorstore = get_vectorstore()

    candidate_docs = vectorstore.similarity_search(user_input, k=30)

    strict_docs = []
    fallback_docs = []

    selected_categories = selected_categories or []

    for doc in candidate_docs:
        skin_ok = doc.metadata.get("skin_type") == selected_skin
        category_ok = (
            True if not selected_categories
            else doc.metadata.get("category") in selected_categories
        )

        if skin_ok and category_ok:
            strict_docs.append(doc)

        if skin_ok:
            fallback_docs.append(doc)

    # Ta bort dubbletter om samma chunk skulle dyka upp flera gånger
    seen_contents = set()
    deduped_strict_docs = []
    for doc in strict_docs:
        if doc.page_content not in seen_contents:
            seen_contents.add(doc.page_content)
            deduped_strict_docs.append(doc)

    seen_contents_fallback = set()
    deduped_fallback_docs = []
    for doc in fallback_docs:
        if doc.page_content not in seen_contents_fallback:
            seen_contents_fallback.add(doc.page_content)
            deduped_fallback_docs.append(doc)

    if deduped_strict_docs:
        final_docs = deduped_strict_docs[:8]
        context_mode = "strict"
    elif deduped_fallback_docs:
        final_docs = deduped_fallback_docs[:8]
        context_mode = "fallback"
    else:
        final_docs = []
        context_mode = "none"

    context = "\n\n".join(doc.page_content for doc in final_docs)

    selected_categories_text = ", ".join(selected_categories) if selected_categories else "inga specifika kategorier"

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Du är en varm, charmig, flirtig och vänskaplig hudvårdsbästis från Kicks. "
            "Du ska låta som en peppig stylist-kompis som ger komplimanger och får användaren att känna sig fabulous. "
            "Svara på svenska, varmt, naturligt och lättsamt.\n\n"

            "VIKTIGT:\n"
            "- Du får ENDAST använda information från kontexten.\n"
            "- Du får inte hitta på produkter, kategorier, priser, länkar eller råd som inte stöds av kontexten.\n"
            "- Om vissa av användarens valda kategorier inte finns i kontexten ska du säga det tydligt men vänligt.\n"
            "- Om exakt rätt kategori saknas men det finns annat relevant för samma hudtyp kan du säga det mjukt.\n"
            "- Presentera inte samma produkt flera gånger.\n"
            "- Om ingen relevant kontext finns ska du säga det tydligt och be användaren förtydliga sitt behov.\n"
            "- Svaret ska kännas levande och trevligt, inte stelt eller robotiskt.\n"
            "- Du får gärna gruppera tipsen efter kategori om flera kategorier valts.\n\n"

            "Användaren har valt hudtyp: {selected_skin}\n"
            "Användaren vill ha tips om: {selected_categories_text}\n"
            "Kontextläge: {context_mode}\n\n"
            "Kontext:\n{context}"
        ),
        ("human", "{input}"),
    ])

    messages = prompt.invoke({
        "context": context if context else "Ingen relevant kontext hittades.",
        "context_mode": context_mode,
        "input": user_input,
        "selected_skin": selected_skin,
        "selected_categories_text": selected_categories_text,
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

if "selected_categories" not in st.session_state:
    st.session_state.selected_categories = []

if "need_product_selection" not in st.session_state:
    st.session_state.need_product_selection = False


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
            inferred_categories = infer_requested_categories(user_input)

            # Om användaren uttryckligen nämner kategorier i texten, uppdatera valet
            if inferred_categories:
                st.session_state.selected_categories = inferred_categories

            if st.session_state.selected_categories:
                response_text = get_rag_response_multi(
                    user_input=user_input,
                    selected_skin=st.session_state.selected_skin,
                    selected_categories=st.session_state.selected_categories,
                )
            else:
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

                followup = get_skin_confirmation_response(choice)
                add_message("assistant", followup)
                st.rerun()
            else:
                st.warning("Välj en hudtyp först.")


if st.session_state.need_product_selection and st.session_state.selected_skin:
    with st.expander("Välj vad du vill ha tips om ✨", expanded=True):
        selected = st.multiselect(
            "Vilka produkter vill du ha hjälp med?",
            PRODUCT_CATEGORIES,
            default=st.session_state.selected_categories
        )

        if st.button("Visa produkttips"):
            if selected:
                st.session_state.selected_categories = selected
                st.session_state.need_product_selection = False

                selected_text = ", ".join(selected[:-1]) + " och " + selected[-1] if len(selected) > 1 else selected[0]

                user_request = (
                    f"Jag vill ha tips för {st.session_state.selected_skin} och är ute efter {selected_text}."
                )

                add_message("user", user_request)

                response_text = get_rag_response_multi(
                    user_input=user_request,
                    selected_skin=st.session_state.selected_skin,
                    selected_categories=selected
                )

                add_message("assistant", response_text)
                st.rerun()
            else:
                st.warning("Välj minst en produktkategori.")