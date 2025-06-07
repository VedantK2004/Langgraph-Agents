from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import Tool
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

google_search = GoogleSerperAPIWrapper()

llm = ChatGoogleGenerativeAI(model="gemma-3-4b-it")

tools = [
    Tool(
        name="Web Search",
        func=google_search.run,
        description="Useful for when you need to answer questions about current events or find information that is not in the training data. Input should be a search query.",
    )
]

template = """
Anser the following questions as best you can. You have access to the following tools:
{tools}
Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do next
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
Begin!
Question: {input}
Thought: {agent_scratchpad}
"""

prompt = PromptTemplate.from_template(template)

search_agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(
    agent=search_agent, tools=tools, verbose=True, return_intermediate_steps=True
)

agent_executor.invoke(
    {
        "input": "Who is the god of cricket? What is the highest score of god of cricket in IPL?",
        "agent_scratchpad": "Cricket",
    }
)
