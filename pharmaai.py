import streamlit as st

def main():
    st.title("ask chatbot!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt=st.chat_input("pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user','content': prompt})

        response="Hi, i am  PharmaAi!"
        st.chat_message('assistant').markdown(response)
        st.session_state.messages.append({'role':'assistant', 'content': response})

if __name__ =="__main__":
    main()