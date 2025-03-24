import streamlit as st
from streamlit_chat import message
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.schema import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationSummaryMemory, ChatMessageHistory

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# Initialize session state variables if they do not exist
if 'llm' not in st.session_state:
    st.session_state['llm'] = ChatGroq(
        model="qwen-2.5-32b",
        temperature=0.3)
if 'conversation' not in st.session_state:
    st.session_state['conversation'] = None
if 'history' not in st.session_state:
    st.session_state['history'] = ChatMessageHistory()
if 'memory' not in st.session_state:
    PromptTemplate(input_variables=['new_lines', 'summary'], template='Progressively summarize the lines of conversation provided, adding onto the previous summary when relevant. If the AI is unable to answer a question, do not add that exchange to the summary. Return only a refined summary.\n\nEXAMPLE\nCurrent summary:\nThe human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good.\n\nNew lines of conversation:\nHuman: Why do you think artificial intelligence is a force for good?\nAI: Because artificial intelligence will help humans reach their full potential.\n\nNew summary:\nThe human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good because it will help humans reach their full potential.\nEND OF EXAMPLE\n\nCurrent summary:\n{summary}\n\nNew lines of conversation:\n{new_lines}\n\nNew summary:')
    st.session_state['memory'] = ConversationSummaryMemory(llm=st.session_state['llm'])

# Setting page title and header
st.set_page_config(page_title="RAG Chatbot", page_icon=":robot_face:")
st.markdown("<h1 style='text-align: center;'>RAG Chatbot Interface </h1>", unsafe_allow_html=True)

def getresponse(userInput):
    # Initialize conversation if it does not exist
    if st.session_state['conversation'] is None:
        llm = st.session_state['llm']
        
        pinecone_api_key = os.environ.get("PINECONE_API_KEY")
        pc = Pinecone(api_key=pinecone_api_key)
        index_name = "vidavox-tech-test"  # change if desired
        index = pc.Index(index_name)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        vector_store = PineconeVectorStore(index=index, embedding=embeddings)
        retriever = vector_store.as_retriever(search_type="mmr")
        
        prompt = PromptTemplate.from_template("""
You are an AI assistant that provides answers strictly based on information from the available tools. Never generate or infer information on your own. Your responses must be polite, professional, and concise.

If the input is a question:

Always use the tools to find the answer.

Never answer based only on chat history.

If the tools do not return relevant information, politely ask the user for clarification.

If the input is not a question:

You may summarize or extract relevant information from the chat history.

Language Matching:

Your response must always be in the same language as the input.

If the user switches languages, adapt accordingly.

Tool Usage Instructions
You have access to the following tools:

{tools}

When using a tool, follow this format:

Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action

When responding to the user, use this format:

Thought: Do I need to use a tool? No
Final Answer: [your response here]

Always use tools to answer questions. Never answer based only on chat history or infer information on your own. Always match the language of the input.

Context
Chat History:
{chat_history}

New Input:
{input}

{agent_scratchpad}

---
                                              """)

        # Build retriever tool
        tool = create_retriever_tool(
            retriever,
            "PDF Information Retriever",
            "This tool extracts and retrieves relevant information from PDF documents about Wakanda based on user queries. It helps quickly summarize and find specific data within PDFs.",
        )
        tools = [tool]

        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)    

        st.session_state['conversation'] = agent_executor

    # Invoke the conversation agent with the user input and chat history
    response = st.session_state['conversation'].invoke({"input": userInput,
                                    "chat_history": st.session_state['memory'].load_memory_variables({})["history"]})    

    # Save the context and update the chat history
    st.session_state['memory'].save_context({"input": userInput}, {"output": response["output"]})
    st.session_state['history'].add_user_message(userInput)
    st.session_state['history'].add_ai_message(response["output"])

# Containers for response and user input
response_container = st.container()
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("Your question goes here:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')

        if submit_button:
            getresponse(user_input)
            print(st.session_state['history'].messages)

            with response_container:
                for i, msg in enumerate(st.session_state['history'].messages):
                    if isinstance(msg, HumanMessage):
                        message(msg.content, is_user=True, key=str(i) + '_user')
                    elif isinstance(msg, AIMessage):
                        message(msg.content, key=str(i) + '_AI')
                    else:
                        pass
