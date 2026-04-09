import streamlit as st

# --- 1. KONFIGURATION ---
st.set_page_config(page_title="Kicks Skin Guide", page_icon="✨")

# --- 2. INITIALISERING AV STATUS (SESSION STATE) ---
if "step" not in st.session_state:
    st.session_state.step = "greeting"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_skin" not in st.session_state:
    st.session_state.selected_skin = None
if "selected_products" not in st.session_state:
    st.session_state.selected_products = []

# --- 3. TITEL ---
st.title("✨ Kicks Skin Guide")
st.subheader("Din personliga hudvårdsbästis")

# --- 4. VISA CHATT-HISTORIK ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 5. HANTERA ANVÄNDARENS INPUT ---
if prompt := st.chat_input("Skriv till din bästis här..."):
    # Spara användarens meddelande
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Botens svar-logik
    with st.chat_message("assistant"):
        if st.session_state.step == "greeting":
            response = "Men hej! Vad roligt att du hör av dig. Jag kan hjälpa dig med allt från en ny rutin till att hitta specifika produkter. Vad funderar du på idag? ✨"
            st.session_state.step = "identify_need"
        
        elif st.session_state.step == "identify_need":
            # Boten berättar om hudtyper (steg 4 i din lista)
            response = (
                "Vad spännande! För att jag ska kunna ge dig de absolut bästa tipsen behöver jag veta din hudtyp. "
                "Huden delas ofta in i:\n"
                "* **Torr hud:** Känns ofta stram och kan flaga.\n"
                "* **Fet hud:** Blir lätt glansig och kan ha förstorade porer.\n"
                "* **Kombinerad hud:** Fet i T-zonen (panna/näsa) men torr på kinderna.\n\n"
                "Vilken av dessa känner du igen dig mest i? Du kan bocka i ditt val nedanför!"
            )
            st.session_state.step = "awaiting_skin_selection"
        
        else:
            response = "Jag väntar på att du ska göra dina val i listan nedan så jag kan ge dig exakta tips! 💖"

        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# --- 6. DYNAMISKA GRÄNSSNITT (Unlockables) ---

# VAL AV HUDTYP (Visas i steg 4 & 5)
if st.session_state.step == "awaiting_skin_selection":
    with st.expander("Välj din hudtyp här", expanded=True):
        skin_choice = st.radio("Min hudtyp:", ["Torr hud", "Fet hud", "Kombinerad hud", "Aknebenägen hud"], index=None)
        if st.button("Bekräfta hudtyp"):
            st.session_state.selected_skin = skin_choice
            st.session_state.step = "skin_confirmed"
            
            # Boten bekräftar valet (steg 5)
            confirm_msg = f"Perfekt! {skin_choice} kräver speciell omtanke. Nu när jag vet det, vilka typer av produkter vill du att jag letar fram åt dig?"
            st.session_state.messages.append({"role": "assistant", "content": confirm_msg})
            st.rerun()

# VAL AV PRODUKTER (Visas först efter hudtyp är vald - steg 5)
if st.session_state.step == "skin_confirmed":
    st.info(f"Vald hudtyp: **{st.session_state.selected_skin}**")
    with st.expander("Vilka produkter letar du efter?", expanded=True):
        prod_choices = st.multiselect("Välj en eller flera:", ["Rengöring", "Ansiktskräm", "Serum", "Ögonkräm", "Solskydd"])
        if st.button("Ge mig rekommendationer! ✨"):
            st.session_state.selected_products = prod_choices
            st.session_state.step = "generate_recommendation"
            st.rerun()

# SLUTGILTIG REKOMMENDATION (Steg 6)
if st.session_state.step == "generate_recommendation":
    with st.chat_message("assistant"):
        loading_msg = f"Håller på att ta fram de bästa alternativen för {st.session_state.selected_skin} inom kategorierna: {', '.join(st.session_state.selected_products)}..."
        st.write(loading_msg)
        
        # --- HÄR KOPPLAR VI PÅ DIN RAG-FUNKTION SENARE ---
        # Exempel på hur svaret kommer se ut:
        final_response = f"Baserat på att du har **{st.session_state.selected_skin}**, rekommenderar jag dessa produkter från kicks.se:\n\n"
        for p in st.session_state.selected_products:
            final_response += f"* **Bästa {p}:** [Produktnamn] - Denna passar dig för att... [Länk till kicks.se]\n"
        
        st.markdown(final_response)
        st.session_state.messages.append({"role": "assistant", "content": final_response})
        
        # Återställ steget om man vill börja om, eller stanna här
        st.session_state.step = "finished"