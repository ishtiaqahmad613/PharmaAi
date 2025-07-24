import streamlit as st

def main():
    st.title("ask chatbot!")

    prompt=st.chat_input("pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)

        response="Hi, i am  pharmaAi!"
        st.chat_message('assistant').markdown(response)

if __name__ =="__main__":
    main()