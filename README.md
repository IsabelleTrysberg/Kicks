RAG Chatbot – Hudvård (Kicks.se)
📌 Projektbeskrivning

Detta projekt implementerar en RAG (Retrieval-Augmented Generation) chatbot specialiserad på hudvård. Chatboten ger tips och produktrekommendationer baserade på innehåll från kicks.se.

Systemet är byggt för att endast använda fördefinierade dokument som kunskapskälla, vilket säkerställer att svaren är relevanta och kontrollerade.

⚙️ Funktionalitet
Chatboten tar input från en Streamlit-app
Identifierar användarens hudtyp:
Fet
Torr
Kombinerad
Hämtar relevanta dokument baserat på:
Hudtyp
Produktkategori
Genererar svar med hjälp av en OpenAI LLM, men endast baserat på inlästa dokument

📂 Datakällor & Struktur
Alla .txt-filer som används i systemet:

Innehåller produktinformation och hudvårdstips från kicks.se
Klassificeras automatiskt baserat på filnamn

⚠️ Viktigt:
Filnamn måste följa en bestämd namnstandard eftersom:

Hudtyp och kategori extraheras direkt från filnamnet
Felaktiga namn → felaktig kategorisering

🧠 Teknologier
Projektet använder följande verktyg:

🔗 langchain, ramverket som kopplar ihop:

Dokument
Vector database
LLM

🤖 langchain-openai 
A
nvänds för att kommunicera med OpenAI:s språkmodeller.

📊 FAISS, en Vector Database som:

Omvandlar text till vektorer (numeriska representationer)
Möjliggör semantisk sökning i dokumenten


🚀 Syfte

Syftet med projektet är att:

Demonstrera hur en RAG-arkitektur fungerar i praktiken
Skapa en användbar chatbot som kombinerar LLM med specifika källor