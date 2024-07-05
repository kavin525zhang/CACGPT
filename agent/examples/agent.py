# url: https://zhuanlan.zhihu.com/p/627948474
import sys, os
sys.path.append(os.getcwd())
from infinity.core.pre_process.llms import WuyaLLM

import re
from typing import List, Union
import textwrap
import time


from langchain.agents import (
    Tool,
    AgentExecutor,
    LLMSingleActionAgent,
    AgentOutputParser
)

from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, LLMChain
from langchain.schema import AgentAction, AgentFinish
from langchain.prompts import PromptTemplate

from langchain.llms.base import BaseLLM

from infinity.data_access.external_apis.model_apis import get_llm_answer
from infinity.controller.schema import QueryContext


MID_OUTPUT = ""

finance_qa_prompt_template = """请根据给定的背景和用户问题，生成一份信息量大、详细、有用并且结构良好的回答，并以Markdown格式呈现。在回答中，请确保清楚地阐述核心观点，并围绕该观点提供充分的细节和例子。

背景：{context}

用户问题：{question}

回答应遵循以下模板：
```markdown
## 背景
提供与问题相关的背景知识，帮助理解问题的上下文。

## 分析
根据问题的不同方面给出具体、详尽的信息和分析。
请使用无序列表(`-` 符号)来组织信息和分析的不同点。

## 结论
简洁明了地总结回答的关键点。
```
请在回答过程中注意逻辑清晰和条理性，确保用户能从回答中获得所需的信息和帮助。不知道答案时可以拒绝回答"""

FINANCE_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=finance_qa_prompt_template,
)

non_finance_qa_prompt_template = """
{context}

结合上述文本，用中文回答问题：“{question}”。"""

NON_FINANCE_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=non_finance_qa_prompt_template,
)

def output_response(response: str) -> None:
    if not response:
        exit(0)
    for line in textwrap.wrap(response, width=60):
        for word in line.split():
            for char in word:
                print(char, end="", flush=True)
                time.sleep(0.1)  # Add a delay of 0.1 seconds between each character
            print(" ", end="", flush=True)  # Add a space between each word
        print()  # Move to the next line after each line is printed
    print("----------------------------------------------------------------")

class FugeDataSource:
    def __init__(self, llm: BaseLLM, 
                 content: str,
                 query_context: QueryContext):
        self.llm = llm
        self.content = content
        self.query_context = query_context

    def generate_non_finance_info(self, query: str) -> str:
        """模拟公司介绍文档数据库，让llm根据抓取信息回答问题"""
        prompt = NON_FINANCE_QA_PROMPT.format(question=query, context=self.content)
        return get_llm_answer(prompt, self.query_context)
    
    def generate_finance_info(self, query: str) -> str: 
        """模拟公司介绍文档数据库，让llm根据抓取信息回答问题"""
        prompt = FINANCE_QA_PROMPT.format(question=query, context=self.content)
        return get_llm_answer(prompt, self.query_context)
    
AGENT_TMPL = """按照给定的格式回答以下问题。你可以使用下面这些工具：

{tools}

回答时需要遵循以下用---括起来的格式：

---
Question: 我需要回答的问题
Thought: 上述这个问题是金融性问题（例如年报、财经等）还是非金融性问题（例如天气、美食等）
Action: ”{tool_names}“ 中的其中一个工具名
Action Input: 选择工具所需要的输入
Observation: 选择工具返回的结果
...（这个思考/行动/行动输入/观察可以重复N次）
Thought: 我现在知道最终答案
Final Answer: 原始输入问题的最终答案
---
    
现在开始回答，记得在给出最终答案前多按照指定格式进行一步一步的推理。

Question: {input}
{agent_scratchpad}
"""

class CustomPromptTemplate(StringPromptTemplate):
    template: str  # 标准模板
    tools: List[Tool]  # 可使用工具集合

    def format(self, **kwargs) -> str:
        """
        按照定义的 template，将需要的值都填写进去。

        Returns:
            str: 填充好后的 template。
        """
        intermediate_steps = kwargs.pop("intermediate_steps")  # 取出中间步骤并进行执行
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
            if re.findall(r"## 背景|## 分析|## 结论", observation):
                global MID_OUTPUT
                MID_OUTPUT = observation.replace("```markdown", "").replace("```", "")
        kwargs["agent_scratchpad"] = thoughts  # 记录下当前想法
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools]
        )  # 枚举所有可使用的工具名+工具描述
        kwargs["tool_names"] = ", ".join(
            [tool.name for tool in self.tools]
        )  # 枚举所有的工具名称
        cur_prompt = self.template.format(**kwargs)
        return cur_prompt
    
class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        """
        解析 llm 的输出，根据输出文本找到需要执行的决策。

        Args:
            llm_output (str): _description_

        Raises:
            ValueError: _description_

        Returns:
            Union[AgentAction, AgentFinish]: _description_
        """
        if "Observation:" not in llm_output and "Final Answer:" in llm_output:  # 如果句子中包含 Final Answer 则代表已经完成
            if MID_OUTPUT:
                return AgentFinish(
                    return_values={"output": MID_OUTPUT},
                    log=llm_output,
                )

            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"  # 解析 action_input 和 action
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)

        return AgentAction(
            tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output
        )
    
def agent_test(user_input, 
               content,
               query_context):
    MID_OUTPUT = ""
    llm = WuyaLLM(model_name="llama2",
                    api_base="http://172.17.120.200:8005",
                    model_kwargs=dict(
                        temperature=0.1,
                        top_p=1.0,
                        top_k=-1,
                        max_new_tokens=2048,
                        presence_penalty=0.3,
                        frequency_penalty=0.3
                    ))

    fuge_data_source = FugeDataSource(llm, 
                                      content,
                                      query_context)
    tools = [
        Tool(
            name="金融类问题问答",
            func=fuge_data_source.generate_finance_info,
            description="拼接金融类问题prompt，供大模型问答",
        ),
        Tool(
            name="非金融类问题问答",
            func=fuge_data_source.generate_non_finance_info,
            description="拼接非金融类问题prompt，供大模型问答",
        ),
    ]
    agent_prompt = CustomPromptTemplate(
        template=AGENT_TMPL,
        tools=tools,
        input_variables=["input", "intermediate_steps"],
    )
    output_parser = CustomOutputParser()

    llm_chain = LLMChain(llm=llm, prompt=agent_prompt)

    tool_names = [tool.name for tool in tools]
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:", "Observation:", "Observation", "Observation:\n", "<|im_end|>", "<im_start>"],
        allowed_tools=tool_names,
    )

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True
    )

    response = agent_executor.run(user_input)
    print("1243445:{}".format(response))
    # return output_response(response)
    return response
    
if __name__ == "__main__":
    # agent_test("上海明天天气如何？", "")
    agent_test("宁德时代2023年营业收入", "有报告显示，其年度营业收入首次突破了4000亿元人民币，达到了4009亿元，同比增长了22.01%。", None)
    ## set api token in terminal
    # llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo")
    # llm = WuyaLLM(model_name="llama2",
    #                            api_base="http://172.17.120.200:8005",
    #                            model_kwargs=dict(
    #                                 temperature=0.1,
    #                                 top_p=1.0,
    #                                 top_k=-1,
    #                                 max_new_tokens=2048,
    #                                 presence_penalty=0.3,
    #                                 frequency_penalty=0.3
    #                             ))

    # fuge_data_source = FugeDataSource(llm)
    # tools = [
    #     Tool(
    #         name="金融类问题问答",
    #         func=fuge_data_source.generate_finance_info,
    #         description="拼接金融类问题prompt，供大模型问答",
    #     ),
    #     Tool(
    #         name="非金融类问题问答",
    #         func=fuge_data_source.generate_non_finance_info,
    #         description="拼接非金融类问题prompt，供大模型问答",
    #     ),
    # ]
    # agent_prompt = CustomPromptTemplate(
    #     template=AGENT_TMPL,
    #     tools=tools,
    #     input_variables=["input", "intermediate_steps"],
    # )
    # output_parser = CustomOutputParser()

    # llm_chain = LLMChain(llm=llm, prompt=agent_prompt)

    # tool_names = [tool.name for tool in tools]
    # agent = LLMSingleActionAgent(
    #     llm_chain=llm_chain,
    #     output_parser=output_parser,
    #     stop=["\nObservation:", "Observation:", "Observation", "Observation:\n", "<|im_end|>", "<im_start>"],
    #     allowed_tools=tool_names,
    # )

    # agent_executor = AgentExecutor.from_agent_and_tools(
    #     agent=agent, tools=tools, verbose=True
    # )

    # while True:
    #     try:
    #         user_input = input("请输入您的问题：")
    #         response = agent_executor.run(user_input)
    #         print("1243445:{}".format(response))
    #         output_response(response)
    #     except KeyboardInterrupt:
    #         break
