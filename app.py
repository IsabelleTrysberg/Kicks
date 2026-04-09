import streamlit as st
from rag_backend import get_rag_response

st.set_page_config(page_title="Kicks Skin Guide", page_icon="✨")

# --- INITIALISERING ---
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hej gorgeous! ✨ Vad kan jag hjälpa dig med?"
    }]

if "show_filters" not in st.session_state:
    st.session_state.show_filters = False

# Standardvärden så variablerna alltid finns
skin_type = None
product_type = None

# --- DYNAMISKA FILTER (Visas bara när det behövs) ---
if st.session_state.show_filters:
    with st.sidebar:
        st.write("### Din profil")

        skin_type = st.selectbox(
            "Bekräfta hudtyp:",
            ["Välj...", "torr-hud", "fet-hud", "kombinerad-hud", "allmänt"],
            index=0
        )

        product_type = st.selectbox(
            "Vad vill du kika på?",
            ["Välj...", "rengöring", "serum", "ansiktskräm", "allmänt"],
            index=0
        )

        if skin_type == "Välj...":
            skin_type = None

        if product_type == "Välj...":
            product_type = None

        if st.button("Uppdatera tips"):
            st.success("Nu vet jag! Jag anpassar mina svar efter detta.")

# --- UI ---
st.title("✨ Kicks Skin Guide")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- CHATT-LOGIK ---
if prompt := st.chat_input("Skriv till din bästis här..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = get_rag_response(
            user_question=prompt,
            selected_skin_type=skin_type,
            selected_category=product_type,
        )

        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

        if "hudtyp" in response.lower() or "vilken typ" in response.lower():
            st.session_state.show_filters = True