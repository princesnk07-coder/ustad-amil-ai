import streamlit as st
import requests  # ✅ yeh missing tha, ab add kar diya

st.set_page_config(page_title="📖 Ustad Amil AI", layout="wide")

st.title("🕌 Ustad Amil AI")
st.write("Apne sawalat likhiye aur Quran-o-Hadith se jawab hasil kijiye (Kanzul Imaan, Hadith, aur Islami Kutub se).")

# User input
question = st.text_input("❓ Sawal likhiye:")

if st.button("🔍 Pucho"):
    if question.strip():
        try:
            response = requests.post(
                "http://127.0.0.1:8000/ask",
                json={"question": question}
            )
            if response.status_code == 200:
                data = response.json()
                st.markdown("### ✅ Jawab:")
                st.write(data["answer"])
                st.markdown(f"📖 **Source:** {data.get('source', 'Unknown')}")
            else:
                st.error("⚠️ Backend error, please check app.py")
        except Exception as e:
            st.error(f"⚠️ Error: {e}")
    else:
        st.warning("Pehle sawal likhiye.")
