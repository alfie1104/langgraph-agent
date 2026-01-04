import os
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

# HuggingFace 사용을 위해 transformers 설치 필요
# HuggingFace에서 받은 모델의 속도 향상을 위해 accelerate, safetensors 설치 필요
# HuggingFace에서 받은 모델의 양자화를 위해 bitsandbytes 설치 필요

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
import torch

load_dotenv()

# #================= OpenAI의 API 활용 시 ==================
# llm = ChatOpenAI(model="gpt-4o")
# #========================================================

#==== Hugging Face를 이용하여 다운받은 local 모델 사용시 ====
model_id = "google/gemma-2-9b-it"
save_path = "./hf_models"

# 8비트 양자화 설정(VRAM 사용량을 줄이기 위해 모델을 압축)
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16 # 계산은 bfloat16으로 하겠다고 명시
)

# 토크나이저를 지정된 폴더로 다운로드
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir = save_path)

# 모델을 지정된 폴더로 다운로드
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    # dtype=torch.bfloat16, # bnb_config안에 데이터 타입을 명시했으므로 quantization_config를 쓸때는 따로 dtype을 안 적어도 됨
    device_map="auto", # "cuda" 대신 auto를 쓰면 효율적임 (GPU 자동 할당)
    cache_dir=save_path
)

# transformers pipeline 생성 (pipeline에 이미 로드된 객체 전달)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

# Langchain LLM 래핑
llm_base = HuggingFacePipeline(pipeline=pipe)

# ChatModel 래핑(LangGraph에서 쓰기 좋게 래핑)
llm = ChatHuggingFace(llm=llm_base)
#========================================================

class AgentState(TypedDict):
    messages : List[Union[HumanMessage , AIMessage]]

def process(state: AgentState) -> AgentState:
    """This node will solve the request you input"""
    response = llm.invoke(state["messages"])

    state["messages"].append(AIMessage(content=response.content)) # AI로 부터 받은 응답을 다시 message 메모리에 넣어줌
    print(f"\nAI: {response.content}")
    # print("CURRENT STATE: ", state["messages"])

    return state

graph = StateGraph(AgentState)
graph.add_node("process",process)
graph.add_edge(START, "process")
graph.add_edge("process",END)
agent = graph.compile()

conversation_history = []

user_input = input("Enter: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages" : conversation_history})
    conversation_history = result["messages"]
    user_input = input("Enter: ")

with open("logging.txt", "w", encoding="utf-8") as file:
    file.write("Your Conversation Log:\n")

    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n")
    file.write("End of Conversation")

print("Conversation saved to logging.txt")