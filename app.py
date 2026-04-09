import streamlit as st

# --- KONFIGURATION AV SIDAN ---
st.set_page_config(page_title="Kicks Skin Guide", page_icon="✨")

# --- TITEL OCH HÄLSNING ---
st.title("✨ Kicks Skin Guide")
st.markdown("### Din stöttande hudvårdsbästis!")

# --- INITIALISERING AV CHATT-MINNE (Session State) ---
# Detta gör att Streamlit kommer ihåg konversationen även när sidan laddas om
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hej gorgeous! Vad har du på hjärtat? ✨"}
    ]

# --- SIDOPANEL FÖR ANVÄNDARENS VAL ---
# Här placerar vi valen för att de inte ska "försvinna" i chatten när den blir lång
with st.sidebar:
    st.header("Din Profil")
    skin_type = st.selectbox(
        "Vilken hudtyp har du?",
        ["Välj hudtyp...", "torr-hud", "fet-hud", "kombinerad-hud"],
        index=0
    )
    
    category = st.selectbox(
        "Vad letar du efter idag?",
        ["Välj kategori...", "rengöring", "serum", "ansiktskräm"],
        index=0
    )

# --- VISA CHATT-HISTORIK ---
# Går igenom alla meddelanden i minnet och skriver ut dem i snygga bubblor
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- HANTERA ANVÄNDARENS INPUT ---
if prompt := st.chat_input("Skriv till din bästis här..."):
    
    # 1. Lägg till användarens meddelande i chatten och minnet
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Logik för att svara (Här kopplar vi på RAG senare)
    with st.chat_message("assistant"):
        # Kontrollera om användaren har valt hudtyp och kategori än
        if skin_type == "Välj hudtyp..." or category == "Välj kategori...":
            response = "I got your back! Men innan jag ger dig mina bästa tips, glöm inte att välja din hudtyp och vad du letar efter i menyn till vänster så det blir helt rätt för just dig! 💖"
        else:
            # Här hamnar vi när användaren har gjort sina val
            response = f"Åh, jag ser att du letar efter {category} för {skin_type}! Låt mig kika i mina anteckningar... (Här kommer RAG-svaret snart!)"
        
        st.markdown(response)
        
    # 3. Spara även assistentens svar i minnet
    st.session_state.messages.append({"role": "assistant", "content": response})