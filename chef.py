from langchain.agents import create_agent
from langchain.messages import HumanMessage
from typing import Dict, Any
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
from langchain.tools import tool
from tavily import TavilyClient

load_dotenv()

tavily = TavilyClient()

@tool
def web_search(query:str)-> Dict[str, Any]:
    """Search the web for information"""

    return tavily.search(query)

SYSTEM_PROMPT = """
Você é um chefe de cozinha, especializado em criar receitas com base em 
ingredientes que o usuário tem na sua cozinha.
Utilize a tool de ´web_search´ para pesquisar na internet receitas para o usuário cozinhar.
Seu objetivo é retornar uma receita para o usuário.

OBSERVAÇÕES:
-NA RECEITA SÓ PODE TER INGREDIENTES QUE O USUÁRIO PASSOU NA QUERY, NÃO INVENTA OU ACRESCENTA
NOVOS INGREDIENTES.
"""

agent = create_agent(
    model="gpt-5-nano",
    tools=[web_search],
    checkpointer=InMemorySaver(),
    system_prompt=SYSTEM_PROMPT,
)

config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [HumanMessage(content="Fala, Chef! Eu tenho macarrão, carne moida e molho de tomate")]},
    config
)

print(response['messages'][-1].content)
