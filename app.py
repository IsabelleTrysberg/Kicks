# Importerar Python-biblioteket random för att kunna välja en slumpmässig hälsningsfras.
import random

# Importerar Streamlit som används för att bygga själva webbappen/gränssnittet.
import streamlit as st

# Importerar OpenAI-chatmodell och embeddings via LangChain.
# ChatOpenAI används för att generera svar.
# OpenAIEmbeddings används för att göra text till vektorer för likhetssökning.
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Importerar FAISS som vektordatabas för att söka bland dokument/chunks.
from langchain_community.vectorstores import FAISS

# Importerar ChatPromptTemplate för att bygga strukturerade prompts till modellen.
from langchain_core.prompts import ChatPromptTemplate


# Sätter sidans titel i browser-tabben och sidans ikon.
# Detta påverkar inte innehållet i själva appen, bara sidinställningarna.
st.set_page_config(page_title="Hudvårdstips från Kicks", page_icon="✨")

# Visar appens huvudrubrik högst upp i gränssnittet.
st.title("✨ Glowie ✨")

# Visar en underrubrik som snabbt förklarar för användaren vad appen gör.
st.markdown("Din hudvårdsbästis som hjälper dig hitta rätt produkter för just din hud 💖")
# Lägger till etisk info 
st.caption("Obs: Informationen är vägledande och ersätter inte professionell hudvårdsrådgivning.")


# Lista med olika hälsningsfraser.
# En av dessa väljs slumpmässigt när appen startar för att ge lite mer personlighet.
GREETINGS = [
    "Hej gorgeous! ✨ Berätta om din hy så hjälper dig hitta en rutin som passar just dig!",
    "Hej där! Jag hjälper dig gärna med din hudvård. Tell me all about it!",
    "What's cooking, good looking? 💖 Hur kan jag hjälpa dig med din hudvård idag?",
    "Hej snygging! ✨ Berätta hur din hud mår så fixar vi en rutin som passar dig perfekt!",
]

# Lista över de hudtyper användaren kan välja mellan i appen.
# Dessa värden används också i filtreringen mot metadata i vektordatabasen.
SKIN_TYPES = ["torr-hud", "fet-hud", "kombinerad-hud"]

# Lista över produktkategorier som användaren kan välja efter hudtyp.
PRODUCT_CATEGORIES = ["rengöring", "serum", "ansiktskräm"]


# Grundprompt som återanvänds i flera funktioner.
# Syftet är att hålla tonen konsekvent i hela appen:
# varm, charmig, peppig, lite flirtig och som en pågående konversation.
BASE_SYSTEM_PROMPT = (
    "Du är en varm, charmig och vänskaplig hudvårdsbästis från Kicks. "
    "Svara på svenska och håll en personlig, peppig och lite flirtig ton.\n\n"
    "VIKTIGT:\n"
    "- Du är mitt i en pågående konversation.\n"
    "- Hälsa inte igen.\n"
    "- Svara naturligt som en fortsättning i en konversation.\n"
)


# Cachear embeddings-objektet så att det inte skapas om varje gång appen körs om.
# Det gör appen snabbare och minskar onödiga API-anrop/initieringar.
@st.cache_resource
def get_embeddings():
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=st.secrets["OPENAI_API_KEY"]
    )


# Cachear vektordatabasen så att FAISS-indexet bara laddas en gång.
# Här läses den lokala databasen "faiss_skincare_v2" in.
@st.cache_resource
def get_vectorstore():
    embeddings = get_embeddings()
    return FAISS.load_local(
        "faiss_skincare_v3",
        embeddings,
        allow_dangerous_deserialization=True
    )


# Cachear språkmodellen så att modellen inte initieras om vid varje interaktion.
# temperature=0.5 gör svaren lite kreativa men fortfarande relativt stabila.
@st.cache_resource
def get_llm():
    return ChatOpenAI(
        model="gpt-4o",
        api_key=st.secrets["OPENAI_API_KEY"],
        temperature=0.5
    )


# Hjälpfunktion som lägger till ett meddelande i chatthistoriken.
# role är t.ex. "user" eller "assistant".
# content är själva texten i meddelandet.
def add_message(role: str, content: str):
    st.session_state.messages.append({"role": role, "content": content})


# Enkel hjälpfunktion som avgör om användarens text verkar handla om hudvård.
# Om något trigger-ord finns i texten returnerar funktionen True.
# Annars False.
# Den används för att styra om boten ska gå in i hudvårdsflödet eller bara småprata.
def user_is_asking_for_skin_help(user_input: str) -> bool:
    text = user_input.lower()
    triggers = [
        "hud", "hudvård", "finnar", "finne", "akne", "torr", "fet", "kombinerad",
        "serum", "rengöring", "ansiktskräm", "kräm", "porer", "glansig",
        "flagar", "rodnad", "hy", "plitor"
    ]
    return any(t in text for t in triggers)


# Genererar ett allmänt svar när användaren inte tydligt frågar om hudvård.
# Här används ingen RAG eller vektordatabas, utan bara språkmodellen.
# Tanken är att Glowie fortfarande ska kunna småprata fritt,
# men samtidigt vänligt styra samtalet mot hudvård.
def get_general_response(user_input: str) -> str:
    llm = get_llm()

    # Skapar en prompt med syfte att styra samtalet in på hudvård + användarens input.
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            BASE_SYSTEM_PROMPT +
            "Din uppgift är att ge tips kring hudvård. Fråga vänligt användaren hur du kan hjälpa hen med sin hudvård. "
            "Du får gärna småprata naturligt och fritt, men du ska alltid tydligt styra tillbaka samtalet mot hudvård. "
            "Om användaren inte beskriver ett hudproblem ska du vänligt påminna hen om att du gärna hjälper till med hudvård. "
            "Ge inte konkreta produkttips i detta steg."
        ),
        ("human", "{input}"),
    ])

    messages = prompt.invoke({"input": user_input})
    return llm.invoke(messages).content


# Analyserar användarens beskrivning av huden för att ge en försiktig tolkning
# av vilken hudtyp det kan röra sig om.
# I detta steg används endast masterfilen om hudtyper.
# Inga konkreta produkttips ska ges här.
def get_skin_type_response(user_input: str) -> str:
    llm = get_llm()
    vectorstore = get_vectorstore()

    # Hämtar de mest liknande dokumentbitarna utifrån användarens fråga.
    candidate_docs = vectorstore.similarity_search(user_input, k=20)

    # Filtrerar fram endast de chunks som kommer från masterfilen om hudtyper.
    docs = [
        doc for doc in candidate_docs
        if doc.metadata.get("source") == "olika-hudtyper-masterguide.txt"
    ][:5]

    # Slår ihop de valda textbitarna till en kontextsträng.
    context = "\n\n".join(doc.page_content for doc in docs)

    # Prompten instruerar modellen att bara använda kontexten,
    # inte hitta på något, och inte ge produkter ännu.
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

    # För in kontext och användarens fråga i prompten.
    messages = prompt.invoke({
        "context": context if context else "Ingen relevant kontext hittades.",
        "input": user_input,
    })

    # Returnerar modellens svar.
    return llm.invoke(messages).content


# Returnerar en statisk hjälpruta som förklarar skillnaden mellan hudtyperna.
# Den visas i UI:t ovanför radioknapparna så användaren får stöd i sitt val.
def build_skin_type_helper_response() -> str:
    return (
        "För att ge dig mina bästa tips behöver du först bekräfta din hudtyp nedan 💖\n\n"
        "**Torr hud** – känns ofta stram, kan flaga och vill ha mycket fukt.\n\n"
        "**Fet hud** – blir lätt glansig, särskilt i T-zonen, och kan få tilltäppta porer.\n\n"
        "**Kombinerad hud** – både och, ofta fetare i panna/näsa/haka men torrare på kinderna.\n\n"
        "Vad känner du mest igen dig i? ✨"
    )


# Används direkt efter att användaren valt hudtyp.
# Den hämtar åter information från masterfilen och bekräftar hudtypen på ett mjukt sätt.
# Den ska guida vidare mot nästa steg: val av produktkategori.
def get_post_skin_selection_response(selected_skin: str) -> str:
    llm = get_llm()
    vectorstore = get_vectorstore()

    # För varje hudtyp finns en separat sökfråga som försöker fånga rätt del av masterfilen.
    query_map = {
        "torr-hud": "torr hud saknar fukt och fett återfuktande produkter rengöring ansiktskräm serum",
        "fet-hud": "fet hud överproduktion av talg hålla huden ren porer rengöring ansiktskräm serum",
        "kombinerad-hud": "kombinerad hud fet t-zon torrare kinder produkter utformade för kombinerad hud rengöring ansiktskräm serum",
    }

    # Söker fram kandidatdokument utifrån vald hudtyp.
    candidate_docs = vectorstore.similarity_search(query_map[selected_skin], k=20)

    # Behåller återigen bara chunks från masterfilen.
    docs = [
        doc for doc in candidate_docs
        if doc.metadata.get("source") == "olika-hudtyper-masterguide.txt"
    ][:5]

    # Bygger kontext.
    context = "\n\n".join(doc.page_content for doc in docs)

    # Prompten instruerar modellen att:
    # - bekräfta vald hudtyp
    # - kort beskriva vad som kännetecknar den
    # - förklara att rengöring, ansiktskräm och serum är en bra grund
    # - men inte ge konkreta produkter ännu
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


# Hämtar svar från vektordatabasen när användaren redan har valt hudtyp
# och sedan skriver vidare i chatten.
# I den förenklade versionen filtreras endast på vald hudtyp.
# Produktkategori tolkas inte längre från användarens fritext, utan väljs i UI:t.
def get_rag_response(user_input: str, selected_skin: str | None) -> str:
    llm = get_llm()
    vectorstore = get_vectorstore()

    # Hämtar relevanta chunks utifrån användarens fråga.
    candidate_docs = vectorstore.similarity_search(user_input, k=20)

    # Här sparas bara dokument som matchar vald hudtyp.
    filtered_docs = []

    for doc in candidate_docs:
        skin_ok = (selected_skin is None or doc.metadata.get("skin_type") == selected_skin)

        if skin_ok:
            filtered_docs.append(doc)

    # Om det finns matchningar används de som kontext.
    # Annars blir kontexten tom.
    if filtered_docs:
        final_docs = filtered_docs[:5]
        context_mode = "skin_only"
    else:
        final_docs = []
        context_mode = "none"

    # Bygger kontext av de dokument som valts.
    context = "\n\n".join(doc.page_content for doc in final_docs)

    # Prompten gör modellen strikt:
    # den får bara använda kontexten och får inte hitta på produkter eller fakta.
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            BASE_SYSTEM_PROMPT +
            "Du måste ENDAST använda information som finns i kontexten. "
            "Du får inte hitta på produkter, kategorier, priser, länkar eller råd som inte stöds av kontexten. "
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


# Hämtar konkreta produktrekommendationer efter att användaren:
# 1) valt hudtyp
# 2) valt en eller flera produktkategorier
#
# Här används endast produktdokument, inte masterfilen.
def get_product_recommendations(selected_skin: str, selected_categories: list[str]) -> str:
    llm = get_llm()
    vectorstore = get_vectorstore()

    # Samlar alla träffade dokument här.
    all_docs = []

    # Söker separat per vald kategori, så att varje kategori får en chans att hitta relevanta dokument.
    for category in selected_categories:
        query = f"{category} {selected_skin} produkttips"

        candidate_docs = vectorstore.similarity_search(query, k=25)

        filtered_docs = []
        for doc in candidate_docs:
            source = doc.metadata.get("source", "")
            skin_ok = doc.metadata.get("skin_type") == selected_skin
            category_ok = doc.metadata.get("category") == category
            is_master = source == "olika-hudtyper-masterguide.txt"

            # Behåller bara dokument som:
            # - matchar rätt hudtyp
            # - matchar rätt kategori
            # - INTE är masterfilen
            if skin_ok and category_ok and not is_master:
                filtered_docs.append(doc)

        # Tar max 5 dokument per kategori för att inte få för lång kontext.
        all_docs.extend(filtered_docs[:5])

    # Tar bort dubbletter så att samma produkttext inte visas flera gånger.
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

    # Bygger slutlig kontext av de unika dokumenten.
    context = "\n\n".join(doc.page_content for doc in unique_docs[:10])

    # Gör om listan av kategorier till läsbar text.
    categories_text = ", ".join(selected_categories)

    # Prompten instruerar modellen att enbart använda produktdokumenten
    # och ta med länk/pris om detta finns i kontexten.
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


# -------------------------
# STATE / SESSION STATE
# -------------------------
# Här sparas data mellan användarens interaktioner i samma session.
# Utan session_state skulle appen "glömma" saker vid varje omkörning.

# Om inga tidigare meddelanden finns i sessionen skapas en ny chatt
# med en slumpmässig hälsningsfras från listan GREETINGS.
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": random.choice(GREETINGS)}
    ]

# Här sparas användarens valda hudtyp.
if "selected_skin" not in st.session_state:
    st.session_state.selected_skin = None

# Styr om UI:t ska visa steget där användaren väljer hudtyp.
if "need_skin_selection" not in st.session_state:
    st.session_state.need_skin_selection = False

# Styr om UI:t ska visa steget där användaren väljer produktkategori.
if "need_product_selection" not in st.session_state:
    st.session_state.need_product_selection = False

# Sparar användarens valda produktkategorier.
if "selected_categories" not in st.session_state:
    st.session_state.selected_categories = []


# Visar tidigare sparade chattmeddelanden på skärmen från session_state.
# Avatar skiljer mellan assistent och användare för bättre UX.
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="✨" if message["role"] == "assistant" else "💬"):
        st.markdown(message["content"])


# Visar chattfältet längst ner.
# Om användaren skickar ett meddelande sparas det i user_input och if-blocket körs.
if user_input := st.chat_input("Skriv till din bästis här..."):
    # Lägger användarens meddelande i chatthistoriken.
    add_message("user", user_input)

    # Visar användarens senaste meddelande direkt i chatten.
    with st.chat_message("user", avatar="💬"):
        st.markdown(user_input)

    # Visar assistentens svarsbubbla.
    with st.chat_message("assistant", avatar="✨"):
        # Visar en laddningsindikator medan modellen söker/genererar svar.
        with st.spinner("Letar fram något fabulous till dig... ✨"):

            # Om användaren redan har valt hudtyp och inte längre är i valet av produktkategori,
            # används det vanliga RAG-flödet på användarens fortsatta fråga.
            # I den förenklade logiken filtreras svaret bara på hudtyp,
            # inte på kategori från användarens fritext.
            if st.session_state.selected_skin and not st.session_state.need_product_selection:
                response_text = get_rag_response(
                    user_input=user_input,
                    selected_skin=st.session_state.selected_skin,
                )
            else:
                # Om meddelandet verkar handla om hudvård triggas hudtypsflödet.
                if user_is_asking_for_skin_help(user_input):
                    response_text = get_skin_type_response(user_input)
                    st.session_state.need_skin_selection = True
                else:
                    # Annars ges ett allmänt svar utan produktlogik.
                    response_text = get_general_response(user_input)

        # Visar svaret i chatten.
        st.markdown(response_text)

        # Sparar också svaret i chatthistoriken.
        add_message("assistant", response_text)


# Om användaren behöver välja hudtyp och ännu inte gjort det
# visas en expander med hjälpinformation + radioknappar.
if st.session_state.need_skin_selection and not st.session_state.selected_skin:
    with st.expander("Välj din hudtyp ✨", expanded=True):
        # Visar statisk hjälpinformation om hudtyper.
        st.markdown(build_skin_type_helper_response())

        # Radioknappar för att välja hudtyp.
        choice = st.radio(
            "Det här känner jag mest igen mig i:",
            SKIN_TYPES,
            index=None
        )

        # När användaren bekräftar sitt val sparas hudtypen i session_state.
        if st.button("Bekräfta hudtyp"):
            if choice:
                st.session_state.selected_skin = choice
                st.session_state.need_skin_selection = False
                st.session_state.need_product_selection = True

                # Efter vald hudtyp får användaren ett bekräftande svar från modellen.
                followup = get_post_skin_selection_response(choice)
                add_message("assistant", followup)
                st.rerun()
            else:
                # Om användaren klickar utan att välja hudtyp visas en varning.
                st.warning("Välj en hudtyp först.")


# Om användaren redan valt hudtyp visas en knapp för att börja om.
if st.session_state.selected_skin:
    if st.button("🔄 Tillbaka till val av hudtyp"):
        # Nollställer vald hudtyp och produktval.
        st.session_state.selected_skin = None
        st.session_state.need_skin_selection = True
        st.session_state.need_product_selection = False
        st.session_state.selected_categories = []

        # Lägger till ett nytt assistentmeddelande som bekräftar omstarten.
        add_message(
            "assistant",
            "Okej babe, vi börjar om ✨ Välj din hudtyp igen så hittar vi rätt vibe för din hud 💖"
        )
        st.rerun()


# Om användaren ska välja produktkategorier visas nästa expander.
if st.session_state.need_product_selection and st.session_state.selected_skin:
    with st.expander("Välj vad du vill ha tips om ✨", expanded=True):
        selected = []
        col1, col2, col3 = st.columns(3)

        # Checkbox för rengöring.
        with col1:
            rengoring = st.checkbox(
                "Rengöring",
                value="rengöring" in st.session_state.selected_categories
            )

        # Checkbox för serum.
        with col2:
            serum = st.checkbox(
                "Serum",
                value="serum" in st.session_state.selected_categories
            )

        # Checkbox för ansiktskräm.
        with col3:
            ansiktskram = st.checkbox(
                "Ansiktskräm",
                value="ansiktskräm" in st.session_state.selected_categories
            )

        # Bygger upp listan selected beroende på vad användaren bockat i.
        if rengoring:
            selected.append("rengöring")
        if serum:
            selected.append("serum")
        if ansiktskram:
            selected.append("ansiktskräm")

        # När användaren klickar på knappen hämtas produkttips.
        if st.button("Visa produkttips"):
            if selected:
                # Sparar valda kategorier i session_state.
                st.session_state.selected_categories = selected
                st.session_state.need_product_selection = False

                # Hämtar produktrekommendationer för vald hudtyp + valda kategorier.
                response_text = get_product_recommendations(
                    selected_skin=st.session_state.selected_skin,
                    selected_categories=selected
                )

                # Sparar svaret i chatten och kör om appen för att visa det.
                add_message("assistant", response_text)
                st.rerun()
            else:
                # Om inget valts visas en varning.
                st.warning("Välj minst en kategori först.")