"""
ReAct (Reasoning and Acting Agent)

# Start -> Agent -> Tools -> Agent -> Tools ... -> Agent -> End

# Annotated - provides additional context without affecting the type itself
# Sequence - To automatically handle the state updates for sequences such as by adding new messages to a chat history

"""

from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage # The foundational class for all message types in LangGraph
from langchain_core.messages import ToolMessage # Passes data back to LLM after it calls a tool such as the content and the tool_call_id
from langchain_core.messages import SystemMessage # Message for providing instructions to the LLM
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# HuggingFace 사용을 위해 transformers 설치 필요
# HuggingFace에서 받은 모델의 속도 향상을 위해 accelerate, safetensors 설치 필요
# HuggingFace에서 받은 모델의 양자화를 위해 bitsandbytes 설치 필요

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
import torch

load_dotenv()


# email = Annotated[str, "This has to be a valid email format!"]

# print(email.__metadata__)

"""
[langgraph.graph.message의 add_messages 함수]
- Reducer function임
- Reducer function?
  . Rules that controls how updates from nodes are combined with the existing state.
  . Tells us how to merge new data into the current state
  . Without a reducer, updates would have replaced the existing value entirely!

    # Without a reducer
    state = {"messages":["Hi!"]}
    update = {"messages":["Nice to meet you!"]}
    new_state = {"messages":["Nice to meet you!"]}

    # With a reducer
    state = {"messages":["Hi!"]}
    update = {"messages":["Nice to meet you!"]}
    new_state = {"messages":["Hi","Nice to meet you!"]}
"""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a: int, b:int):
    """This is an addition function that adds 2 numbers together"""
    return a + b

@tool
def subtract(a: int, b:int):
    """Subtraction function"""
    return a - b

@tool
def multiply(a: int, b:int):
    """Multiplication function"""
    return a * b

tools = [add, subtract, multiply]

# #================= OpenAI의 API 활용 시 ==================
llm = ChatOpenAI(model = "gpt-4o").bind_tools(tools)
# #========================================================


# Local LLM 모델은 OpenAI 스타일의 구조화된 툴 콜링을 지원하지 않아서 오류가 발생할 수 있음
# #==== Hugging Face를 이용하여 다운받은 local 모델 사용시 ====
# model_id = "google/gemma-2-9b-it"
# save_path = "./hf_models"

# # 8비트 양자화 설정(VRAM 사용량을 줄이기 위해 모델을 압축)
# bnb_config = BitsAndBytesConfig(
#     load_in_8bit=True,
#     bnb_4bit_compute_dtype=torch.bfloat16 # 계산은 bfloat16으로 하겠다고 명시
# )

# # 토크나이저를 지정된 폴더로 다운로드
# tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir = save_path)

# # 모델을 지정된 폴더로 다운로드
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     quantization_config=bnb_config,
#     device_map="auto",
#     cache_dir=save_path
# )

# # transformers pipeline 생성 (pipeline에 이미 로드된 객체 전달)
# pipe = pipeline(
#     "text-generation",
#     model = model,
#     tokenizer=tokenizer
# )

# # Langchain LLM 래핑
# llm_base = HuggingFacePipeline(pipeline=pipe)

# # ChatModel 래핑(LangGraph에서 쓰기 좋게 래핑)
# llm = ChatHuggingFace(llm = llm_base).bind_tools(tools)
# #========================================================


def model_call(state:AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content = "You are my AI assistant, please answer my query to the best of your ability."
    )
    response = llm.invoke([system_prompt] + state["messages"])

    # messages는 reducer인 add_messages로 정의되어있으므로 아래와 같이 신규 response를 할당하면 알아서 기존 데이터에 덧붙여짐
    return {"messages":[response]}

# Loop에서 loop를 계속 돌지 결정하는 조건
def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        # 더 이상 호출할 도구가 없을 경우 "end"를 리턴
        return "end"
    else:
        return "continue"

graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue":"tools",
        "end":END
    }
)

graph.add_edge("tools","our_agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

# inputs = {"messages": [("user","Add 34 + 21. Add 3 + 4. Add 12 + 12")]}
# inputs = {"messages": [("user","Add 40 + 12 and then multiply the result by 6")]}
inputs = {"messages": [("user","Add 40 + 12 and then multiply the result by 6. Also tell me a joke please")]}
print_stream(app.stream(inputs, stream_mode="values"))