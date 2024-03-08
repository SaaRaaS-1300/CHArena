# 导入父目录包
import sys
import os

# 搜寻路径
dir = os.path.dirname(os.path.abspath(__file__))
# 将父目录添加到 Python 的搜索路径中
sys.path.append(os.path.dirname(os.path.dirname(dir)))

# import Xxx
from build.personas import RoleplayerModel
from typing import Dict, List, Optional, Tuple, Union
from lagent.agents.internlm2_agent import Internlm2Protocol
from lagent.schema import AgentStatusCode, AgentReturn
from lagent.llms.meta_template import INTERNLM2_META
from lagent.actions.base_action import BaseAction
from lagent.agents.base_agent import BaseAgent
from lagent.actions import ActionExecutor


class RoleplayerAgentI2(BaseAgent):
    """
    RoleplayerAgentI2 is based on BaseAgent.

    Args:
        llm (BaseModel): 
            the language model.
        action_executor (ActionExecutor): 
            the action executor.
        protocol (object): 
            the protocol of the agent, which is used to
            generate the prompt of the agent and parse the 
            response fromthe llm.
    """

    def __init__(self,
                 llm: RoleplayerModel,
                 plugin_executor: ActionExecutor = None,
                 interpreter_executor: ActionExecutor = None,
                 protocol=Internlm2Protocol(),
                 max_turn: int = 3) -> None:
        self.max_turn = max_turn
        self._interpreter_executor = interpreter_executor
        super().__init__(
            llm=llm, 
            action_executor=plugin_executor, 
            protocol=protocol
        )

    def chat(self, message: Union[str, Dict], history: List[Dict] = None, **kwargs) -> AgentReturn:
        """
        Generate a response to the user's message based on the chat history.

        Args:
            message (str):
                the user's message.
            history (List[Dict[str, str]], optional):
                the chat history, where each element is a dictionary
                with keys 'role' (either 'user' or 'assistant') and 'content'
                (the text content of the message). Defaults to None.

        Returns:
            AgentReturn: an object containing the response and other metadata.
        """
        # 检测
        if history is None:
            history = []

        # 构造
        if isinstance(message, str):
            message = dict(role='user', content=message)
        if isinstance(message, dict):
            message = [message]
        history.extend(message)
        agent_return = AgentReturn()

        # max turn for chat
        for _ in range(self.max_turn):
            # Step 1. 格式化消息
            prompt = self._protocol.format(inner_step=history, **kwargs)
            response = self._llm.chat(prompt)

            print("消息原型: ", response)
            # Step 2. 处理返回消息
            # 解析模型的响应，提取纯文本消息
            split_msg = response.split("assistant\n")

            # 去除前面的部分，只保留最后一个 assistant 之后的消息
            response = split_msg[-1].split("<|im_end|>")[0]

            # Step 3. 置入消息结果
            history.append(dict(role='assistant', content=response))
            agent_return.response = response
            agent_return.inner_steps = history
            agent_return.state = AgentStatusCode.END
            break

        return agent_return
        