import json
from pathlib import Path

from click import prompt
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
from langchain_core.runnables import RunnableLambda
from pandas.core.config_init import reader_engine_doc

from search import query
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import  StrOutputParser


class Robot:
    with open('config.json', 'r') as f:
        config = json.load(f)
    sys_msg = config['system']
    human_msg = config['human']
    def __init__(self,query_content):
        self.model = ChatOllama(model='qwen3:8b')
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.sys_msg),
            ("human", self.human_msg),
        ])
        self.query=RunnableLambda(query)
        self.query_content=query_content
        self.memory_path=Path('memory\\memory.txt')
    def ask(self):
        content=self.query_content
        chain=self.query|self.prompt|self.model|StrOutputParser()
        return chain.invoke(input={"query_content":self.query_content})

    def memory(self):
        with open(self.memory_path,'r') as f:
            reader=f.read()
        if reader=='':
            return None
        else:
            return eval(reader)

    def human(self):
        memory_loader=self.memory()
        if memory_loader==None:
            message_history = []
            human_ask = input('请输入你的病症：\n')
            self.query_content=human_ask
            answer=self.ask()
            message_history.append(("system", self.sys_msg))
            message_history.append(("human", human_ask))
            message_history.append(("ai",answer))
            with open('memory\\memory.txt','w') as f:
                f.write(str(message_history))
            memory_loader=self.memory()
        while memory_loader:
            human_ask = input()
            message_history=memory_loader
            message_history.append(("human",human_ask))
            CPT = ChatPromptTemplate.from_messages(message_history)
            chain= CPT|self.model
            ai_ans=chain.invoke(input={})
            message_history.append(("ai",ai_ans))
        with open('memory\\memory.txt','w') as f:
            f.write(str(message_history))
robot=Robot(query_content='')
robot.human()
