# 라이브러리 로딩

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder     
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory  
from langchain_google_genai import ChatGoogleGenerativeAI     
from langchain_core.output_parsers import StrOutputParser     
from dotenv import load_dotenv    

load_dotenv()              


# ai 모델을 만든다
llm_model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.5)


# 프롬프트 설정을 한다.
prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 질문을 하는 human의 아내야"),             
    MessagesPlaceholder(variable_name="history"),                    
    ("human", "{question}")                                          
])


# chain 실행기를 만든다
chain = prompt | llm_model  | StrOutputParser()


# 대화 히스토리를 저장할(history에 너을) 메모리 객체를 만든다
store = {}     # 대화 히스토리를 저장할 딕셔너리 객체

def get_message_history(session_id: str):
    """세션 ID에 해당하는 메모리 기반 대화 히스토리 객체를 반환하는 함수"""
    
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
            
    return store[session_id]


# chain 을 메모리 기반 실행기로 래핑한다
with_memory_chain = RunnableWithMessageHistory(chain,                               
                                               get_message_history,                 
                                               input_messages_key="question",       
                                               history_messages_key="history")      


##################### 여기서부터 대화가 cli 방식으로 계속 이어질 수 있도록 한다 #####################

def chat_cli():
    """CLI 방식의 채팅 인터페이스 함수"""
    
    print("Gemini 챗봇에 오신 것을 환영합니다! 종료하려면 'exit'를 입력하세요.")
    
    session_id = input("세션 이름을 입력하세요 (예: user1): ")
    
    print("-" * 60)

    while True:
        
        user_input = input(f"\n{session_id}: ")   # 사용자 입력 받기

        if user_input.lower() == "exit":
            print("채팅을 종료합니다. 안녕히 가세요!")
            break            # 반복문 자체를 종료함

        if not user_input.strip():
            print("뭐라도 넣어야 ai가 응답을 하지 ㅠㅠ")
            continue         # 반복문을 끝내고 처음부터 다시 시작
        
        
        # 메모리 기반 체인을 사용하여 응답 생성
        # 에러가 날 경우를 대비하여 try except 문으로 감싸는 방법이 좋음.(ai 에러 수시로 발생함, 파싱 문제도 있고)

        try:
            response = with_memory_chain.invoke({"question": user_input},
                                                config={"configurable": {"session_id": session_id}})
            
            print(f"아내: {response}")   # Gemini의 응답 출력
        except Exception as e:
            print(f"오류가 발생했습니다: {e}")

if __name__ == "__main__":
    chat_cli()   # 채팅 CLI 함수 실행