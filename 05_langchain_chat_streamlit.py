# 라이브러리 로딩

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder     
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory  
from langchain_google_genai import ChatGoogleGenerativeAI     
from langchain_core.output_parsers import StrOutputParser     
from dotenv import load_dotenv    

import streamlit as st   # 스트림릿 모듈 추가 pip install streamlit 해서 설치 해줘야되고 requirements 에도 추가해준다

load_dotenv()       

# 타이틀 하나 너어준다
st.set_page_config(page_title="제미나이 채팅")
st.title("랭체인 제미나이 채팅 애플리케이션")

# 이런저런 세팅을 하기 위하여 사이드바 하나를 만들어 놓는다
with st.sidebar:
    st.header("⚙️설정")
    
    # 시스템 인스트럭션
    # 제목
    st.markdown("### System Insruction(AI 혁할 및 지침)")
    
    # 여기에 글씨를 넣으면 system 으로 프롬프트에 들어간다
    system_instruction = st.text_area("",
                                      # value="너는 제미나이 AI야",                  # 기본값
                                      placeholder="예 : 너는 파이썬 선생님이야!",
                                      label_visibility="collapsed")

    # 답변 길이에 관한 부분
    
    st.markdown("### 답변 길이")
    lenth_option = ["100자 내외", "500자 내외", "1000자 내외", "제한없음"]
    lenth_value = st.select_slider("", lenth_option, label_visibility="collapsed", value="1000자 내외")

    if lenth_value != "제한없음":
        system_instruction = system_instruction + f"답변은 {lenth_value} 정도로 해 줘!"
    else:
        system_instruction = system_instruction + "답변은 상세하게 길이에 상관없이 상세하게 해 줘" 


    # 창의성 슬라이더 하나 넣는다
    st.markdown("### Temperature(창의성)")
    # 슬라이더 설정: (이름, 최솟값, 최댓값, 기본값, 간격)
    temperature_config = st.slider("", 0.0, 2.0 , 1.0, 0.1, label_visibility="collapsed")
    st.write('''창의성에 관한 부분입니다. 
             높으면 창의력이 많아지고 작으면 
             단순명료하게 대답합니다.''')  


# 스트림릿은 페이지를 자꾸자꾸 다시 읽어오기 때문에 다시읽어오지 못하도록 세션 스테이트 안에 
# 안바뀌어야 하는 내용을 넣어줘야 한다
# history 가 저장되는 store 딕셔너리는 리로딩 될때만다 초기화 되면 안되니
# session_state 안에 넣어 놓는다... 즉 없을때만 만든다
if "store" not in st.session_state:
    st.session_state.store = {}

# 대화를 보낼 내용의 메세지도 화면에 보여야 되니까 session_state
if "messege" not in st.session_state:
    st.session_state.messege = []
    

def get_message_history(session_id: str):
    """세션 ID에 해당하는 메모리 기반 대화 히스토리 객체를 반환하는 함수"""
    
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = InMemoryChatMessageHistory()
            
    return st.session_state.store[session_id]   


# ai 모델을 만든다
llm_model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=temperature_config)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_instruction),             
    MessagesPlaceholder(variable_name="history"),                    
    ("human", "{question}")                                          
])

# chain 실행기를 만든다
chain = prompt | llm_model  | StrOutputParser()

# chain 을 메모리 기반 실행기로 래핑한다
with_memory_chain = RunnableWithMessageHistory(chain,                               
                                               get_message_history,                 
                                               input_messages_key="question",       
                                               history_messages_key="history") 

# 세션스테이트에 저장된 내용을 읽어서 chat_messege 오브젝트(아마 창일껏임)에 써준다
for messege in st.session_state.messege:
    with st.chat_message(messege["role"]):
        st.write(messege["content"])

# 아래 둘중 하나만 하면 된다
# user_input = st.chat_input("질문이나 대화 내용을 입력해 주세요" ) # 인풋 오브젝트 만들고

# if user_input:              # 그 오브젝트에 출력
#     print(user_input)

if user_input := st.chat_input("질문이나 대화 내용을 입력해 주세요"): # 입력창에 글씨를 쓰고 입력 하면 이라는 말
    # 여기서부터 메세지 리스트에 대화내용을 넣고 메세지가 윗쪽 칸에 나온다
    # 당연하지만 store 에도 대화 내용을 추가한다.

    # 일단 메세지 표시부터 해보자
    with st.chat_message("user"):   # 챗 메세지 오브젝트의 "user 란 key 값에"
        st.write(user_input)        # 사용자가 입력한 인풋을 적어준다

    # 이제 세션 스테이트 메세지에다가 지금 올린 내용을 추가해 준다
    st.session_state.messege.append({"role": "user", "content": user_input})  

    # AI 의 응답을 받는 로직을 만든다
    with st.chat_message("ai"):                     # 대화내용에 너을 창을 열어서
        with st.spinner("AI 가 생각 중입니다...."):   # ai 가 답변할때까지 기다린다
            try:   
                # ai 의 응답을 받는다
                response =with_memory_chain.invoke({"question" : user_input},
                                                config={"configurable" : {"session_id" : "GEMINI"}})
                st.write(response)
                st.session_state.messege.append({"role": "ai", "content": response})
            
            except Exception as e:
                st.error(e)