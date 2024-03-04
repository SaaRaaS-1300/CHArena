# 基础包
from typing import Dict, List, Optional, Union
from copy import deepcopy
import logging
import json

# Lagent 相关包
from lagent.schema import ActionReturn, ActionStatusCode, AgentReturn, AgentStatusCode, ModelStatusCode
from lagent.agents.base_agent import BaseAgent
from lagent.llms import BaseModel
from lagent.actions import ActionExecutor

# 设定参数
META_CN = ('当可以启用工具时，请根据需求调用合适的工具')
PLUGIN_CN = (
    '你可以使用如下工具：'
    '\n{prompt}\n'
    '你需要合理应用这些工具，避免不必要的工具调用! '
    '同时，禁止你捏造聊天内容，或者胡乱调用工具！'
)


"""
    Roleplayer Interpreter Prompt 专用于规定角色名称
"""
class RoleplayerInterpreterPrompt:
    
    def __init__(
        self,
        role_name: str = "默认玩家"
    ):
        self.role_name = role_name

    def build_interpreter_prompt(self):
        # 构造 Interpreter Prompt
        INTERPRETER_CN = (
           f'你是{self.role_name}，正在参与角色聊天能力竞赛。'
            '你具有调用工具的能力，这些工具将为你在聊天时提供帮助。'
            '你需要按照自己的角色风格与其他人进行沟通，并保持逻辑严谨！'
        )
        return INTERPRETER_CN


"""
    Roleplayer Protocol 专用于设计角色扮演模型规则
"""
class RoleplayerProtocol:

    # 初始化RoleplayerProtocol类
    def __init__(
        self,
        # 元提示，用于描述角色扮演的场景
        meta_prompt: str = META_CN,  
        # 解释器提示，用于描述解释器的功能
        interpreter_prompt: str = RoleplayerInterpreterPrompt().build_interpreter_prompt(), 
        # 插件提示，用于描述插件的功能
        plugin_prompt: str = PLUGIN_CN,  
        # few-shot学习的示例对话
        few_shot: Optional[List] = None,  
        language: Dict = dict(  # 语言配置
            begin='',  # 语言开始标记
            end='',  # 语言结束标记
            belong='assistant',  # 语言所属角色
        ),
        tool: Dict = dict(  # 工具配置
            begin='{start_token}{name}\n',  # 工具使用开始标记
            start_token='<|action_start|>',  # 工具使用开始标记的起始符号
            name_map=dict(plugin='<|plugin|>', interpreter='<|interpreter|>'),  # 工具名称映射
            belong='assistant',  # 工具所属角色
            end='<|action_end|>\n',  # 工具使用结束标记
        ),
        execute: Dict = dict(  # 执行配置
            role='execute',  # 执行角色
            begin='',  # 执行开始标记
            end='',  # 执行结束标记
            fallback_role='environment',  # 回退角色
        )
    ) -> None:
        self.meta_prompt = meta_prompt  # 保存元提示
        self.interpreter_prompt = interpreter_prompt  # 保存解释器提示
        self.plugin_prompt = plugin_prompt  # 保存插件提示
        self.roles_cfg = dict(tool=tool, language=language)  # 保存工具和语言配置
        self.language = language  # 保存语言配置
        self.execute = execute  # 保存执行配置
        self.tool = tool  # 保存工具配置
        self.few_shot = few_shot  # 保存few-shot学习的示例对话

    # 格式化子角色消息
    def format_sub_role(self, messages: List[Dict]) -> List[Dict]: 

        def format_interpreter(message):  # 格式化解释器消息
            if isinstance(message['content'], dict):  # 判断消息内容是否为字典
                return dict(  # 返回格式化后的消息
                    role=message['role'],  # 角色
                    name=message['name'],  # 名称
                    content=message['content']['parameters']['command']  # 命令
                )
            else:  # 消息内容不是字典
                return message  # 返回原消息

        def format_plugin(message):  # 格式化插件消息
            if isinstance(message['content'], dict):  # 判断消息内容是否为字典
                return dict(  # 返回格式化后的消息
                    role=message['role'],  # 角色
                    name=message['name'],  # 名称
                    content=json.dumps(message['content'])  # 内容为json格式的字符串
                )
            else:  # 消息内容不是字典
                return message  # 返回原消息

        new_message = list()  # 新消息列表
        for message in messages:  # 遍历消息列表
            if message['role'] in [  # 判断消息角色是否在以下列表中
                    'assistant', 'user', 'system', 'environment'
            ]:
                new_message.append(message)  # 添加到新消息列表中
                continue  # 继续遍历下一条消息
            role_cfg = self.roles_cfg[message['role']]  # 获取角色配置
            begin = role_cfg['begin']  # 获取角色开始标识
            if message['role'] == 'tool':  # 判断消息角色是否为'tool'
                if message['name'] == 'interpreter':  # 判断消息名称是否为'interpreter'
                    message = format_interpreter(message)  # 格式化解释器消息
                elif message['name'] == 'plugin':  # 判断消息名称是否为'plugin'
                    message = format_plugin(message)  # 格式化插件消息
                else:  # 消息名称不是'interpreter'或'plugin'
                    raise NotImplementedError  # 抛出未实现错误
                begin = role_cfg['begin'].format(  # 格式化角色开始标识
                    start_token=role_cfg.get('start_token', ''),  # 获取开始标记
                    name=role_cfg.get('name_map', {}).get(message['name'], '')  # 获取名称映射
                )
            new_content = begin + message['content'] + role_cfg['end']  # 组合新消息内容
            if role_cfg.get('fallback_role'):  # 判断是否存在回退角色
                new_message.append(  # 添加到新消息列表中
                    dict(role=role_cfg['fallback_role'], content=new_content)  # 创建新消息
                )
            elif role_cfg.get('belong'):  # 判断是否存在所属角色
                if new_message[-1]['role'] != role_cfg.get('belong'):  # 判断上一条消息的角色是否与所属角色一致
                    new_message.append(  # 添加到新消息列表中
                        dict(role=role_cfg.get('belong'), content=new_content)  # 创建新消息
                    )
                else:  # 上一条消息的角色与所属角色一致
                    new_message[-1]['content'] += new_content  # 将新消息内容追加到上一条消息中
            else:  # 不存在回退角色或所属角色
                new_message.append(  # 添加到新消息列表中
                    dict(role=message['role'], content=new_content)  # 创建新消息
                )

        # 返回新消息列表
        return new_message  

    # 定义一个名为format的函数，带有self参数
    def format(self,  
               inner_step: List[Dict],  # 输入参数：一个列表，包含多个字典，每个字典表示一条消息
               plugin_executor: ActionExecutor = None,  # 输入参数：一个ActionExecutor类型的对象，表示插件执行器，默认值为None
               interpreter_executor: ActionExecutor = None,  # 输入参数：一个ActionExecutor类型的对象，表示解释器执行器，默认值为None
               **kwargs) -> list:  # 输入参数：可变参数，返回值：一个列表

        formatted = []  # 初始化一个空列表，用于存储格式化后的消息
        # 如果有元提示
        if self.meta_prompt: 
            formatted.append(dict(role='system', content=self.meta_prompt))  # 将元提示添加到格式化后的消息列表中
        if interpreter_executor and self.interpreter_prompt:  # 如果有解释器执行器和解释器提示
            interpreter_info = interpreter_executor.get_actions_info()[0]  # 获取解释器的动作信息
            interpreter_prompt = self.interpreter_prompt.format(  # 格式化解释器提示
                code_prompt=interpreter_info['description'])
            formatted.append(  # 将格式化后的解释器提示添加到格式化后的消息列表中
                dict(
                    role='system',
                    content=interpreter_prompt,
                    name='interpreter'))

        # 如果有插件执行器、插件动作和插件提示
        if plugin_executor and plugin_executor.actions and self.plugin_prompt: 
            plugin_descriptions = []  # 初始化一个空列表，用于存储插件描述
            for api_info in plugin_executor.get_actions_info():  # 遍历插件执行器的动作信息
                plugin = deepcopy(api_info)  # 深拷贝动作信息
                if isinstance(api_info, dict):  # 如果动作信息是一个字典
                    tool_name = api_info['name'].split('.')[0]  # 获取工具名称
                    plugin['description'] = API_PREFIX.format(  # 格式化插件描述
                        tool_name=tool_name, description=plugin['description'])
                plugin_descriptions.append(plugin)  # 将插件描述添加到插件描述列表中
            plugin_prompt = self.plugin_prompt.format(  # 格式化插件提示
                prompt=json.dumps(
                    plugin_descriptions, ensure_ascii=False, indent=4))
            formatted.append(  # 将格式化后的插件提示添加到格式化后的消息列表中
                dict(role='system', content=plugin_prompt, name='plugin'))

        if self.few_shot:  # 如果有少量样本
            for few_shot in self.few_shot:  # 遍历少量样本
                formatted += self.format_sub_role(few_shot)  # 格式化少量样本并添加到格式化后的消息列表中

        # 格式化输入的消息列表并添加到格式化后的消息列表中
        formatted += self.format_sub_role(inner_step) 
        return formatted  # 返回格式化后的消息列表

    def parse(self, message, plugin_executor: ActionExecutor, interpreter_executor: ActionExecutor):
        # 解析消息，判断消息是插件还是解释器的输出，并提取相应的动作和参数
        if self.language['begin']:
            # 如果消息开头有特定的语言标记，则去除该标记
            message = message.split(self.language['begin'])[-1]
        if self.tool['name_map']['plugin'] in message:
            # 如果消息中包含插件名称，则将消息分割成插件名称和动作
            message, action = message.split(
                f"{self.tool['start_token']}{self.tool['name_map']['plugin']}")
            action = action.split(self.tool['end'].strip())[0]
            # 返回插件名称、消息和动作
            return 'plugin', message, action
        if self.tool['name_map']['interpreter'] in message:
            # 如果消息中包含解释器名称，则将消息分割成解释器名称和代码
            message, code = message.split(
                f"{self.tool['start_token']}"
                f"{self.tool['name_map']['interpreter']}")
            code = code.split(self.tool['end'].strip())[0].strip()
            # 返回解释器名称、消息和代码
            return 'interpreter', message, dict(
                name=interpreter_executor.action_names()[0],
                parameters=dict(command=code))
        # 如果消息中不包含插件或解释器名称，则返回None、消息和None
        return None, message.split(self.tool['start_token'])[0], None

    def format_response(self, action_return, name) -> dict:
        # 格式化动作返回值，生成响应消息
        if action_return.state == ActionStatusCode.SUCCESS:
            # 如果动作执行成功，则提取动作返回值
            response = action_return.format_result()
        else:
            # 如果动作执行失败，则提取动作错误信息
            response = action_return.errmsg
        # 生成响应消息内容
        content = self.execute['begin'] + response + self.execute['end']

        if self.execute.get('fallback_role'):
            # 如果执行配置中有回退角色，则使用回退角色
            return dict(
                role=self.execute['fallback_role'], content=content, name=name)
        elif self.execute.get('belong'):
            # 如果执行配置中有所属角色，则使用所属角色
            return dict(
                role=self.execute['belong'], content=content, name=name)
        
        # 如果执行配置中没有回退角色和所属角色，则使用默认角色
        return dict(role=self.execute['role'], content=response, name=name)


"""
    Roleplayer Agent 专用于设计角色扮演模型 Agent 的对象
"""
class RoleplayerAgent(BaseAgent):

    def __init__(self,
                 llm: BaseModel,
                 plugin_executor: ActionExecutor = None,
                 interpreter_executor: ActionExecutor = None,
                 protocol=RoleplayerProtocol(),
                 max_turn: int = 3) -> None:
        self.max_turn = max_turn
        self._interpreter_executor = interpreter_executor
        super().__init__(
            llm=llm, 
            action_executor=plugin_executor, 
            protocol=protocol
        )
    
    def chat(self, message: Union[str, Dict], **kwargs) -> AgentReturn:
        # 如果消息是字符串，转换为字典格式
        if isinstance(message, str):
            message = dict(role='user', content=message)
        # 如果消息是字典，转换为列表格式
        if isinstance(message, dict):
            message = [message]
        # 保存对话历史记录
        inner_history = message[:]
        # 设置初始回复位置
        offset = len(inner_history)
        # 初始化 AgentReturn 对象
        agent_return = AgentReturn()
        # 循环最大回合数
        for _ in range(self.max_turn):
            # 格式化提示信息
            prompt = self._protocol.format(
                inner_step=inner_history,
                plugin_executor=self._action_executor,
                interpreter_executor=self._interpreter_executor,
            )
            # 调用语言模型生成回复
            response = self._llm.chat(prompt, **kwargs)
            # 解析回复信息
            name, language, action = self._protocol.parse(
                message=response,
                plugin_executor=self._action_executor,
                interpreter_executor=self._interpreter_executor,
            )
            # 判断回复类型
            if name:
                # 如果是插件回复
                if name == 'plugin':
                    # 获取插件执行器
                    if self._action_executor:
                        executor = self._action_executor
                    else:
                        logging.info(msg='No plugin is instantiated!')
                        continue
                    # 解析动作参数
                    try:
                        action = json.loads(action)
                    except Exception as e:
                        logging.info(msg=f'Invaild action {e}')
                        continue
                # 如果是解释器回复
                elif name == 'interpreter':
                    # 获取解释器执行器
                    if self._interpreter_executor:
                        executor = self._interpreter_executor
                    else:
                        logging.info(msg='No interpreter is instantiated!')
                        continue
                # 其他回复类型
                else:
                    logging.info(
                        msg=(f"Invalid name '{name}'. Currently only 'plugin' "
                             "and 'interpreter' are supported."))
                    continue
                # 执行动作
                action_return: ActionReturn = executor(action['name'],
                                                       action['parameters'])
                # 记录动作思路
                action_return.thought = language
                # 添加到 AgentReturn 对象中
                agent_return.actions.append(action_return)
            # 添加到对话历史记录中
            inner_history.append(dict(role='language', content=language))
            # 判断是否结束对话
            if not name or action_return.type == executor.finish_action.name:
                # 设置回复信息
                agent_return.response = language
                # 设置状态为结束
                agent_return.state = AgentStatusCode.END
                break
            else:
                # 添加到对话历史记录中
                inner_history.append(
                    dict(role='tool', content=action, name=name))
                inner_history.append(
                    self._protocol.format_response(action_return, name=name))
        # 设置内部步骤
        agent_return.inner_steps = inner_history[offset:]
        # 返回 AgentReturn 对象
        return agent_return

    def stream_chat(self, message: List[dict], **kwargs) -> AgentReturn:
        if isinstance(message, str):
            message = dict(role='user', content=message)
        if isinstance(message, dict):
            message = [message]
        inner_history = message[:]
        offset = len(inner_history)
        agent_return = AgentReturn()
        last_agent_state = AgentStatusCode.SESSION_READY
        for _ in range(self.max_turn):
            # list of dict
            prompt = self._protocol.format(
                inner_step=inner_history,
                plugin_executor=self._action_executor,
                interpreter_executor=self._interpreter_executor,
            )
            response = ''
            for model_state, res, _ in self._llm.stream_chat(prompt, **kwargs):
                model_state: ModelStatusCode
                response = res
                if model_state.value < 0:
                    agent_return.state = getattr(AgentStatusCode,
                                                 model_state.name)
                    yield deepcopy(agent_return)
                    return
                else:
                    name, language, action = self._protocol.parse(
                        message=response,
                        plugin_executor=self._action_executor,
                        interpreter_executor=self._interpreter_executor,
                    )
                    if name:
                        if model_state == ModelStatusCode.END:
                            agent_state = last_agent_state + 1
                            if name == 'plugin':
                                if self._action_executor:
                                    executor = self._action_executor
                                else:
                                    logging.info(
                                        msg='No plugin is instantiated!')
                                    continue
                                try:
                                    action = json.loads(action)
                                except Exception as e:
                                    logging.info(msg=f'Invaild action {e}')
                                    continue
                            elif name == 'interpreter':
                                if self._interpreter_executor:
                                    executor = self._interpreter_executor
                                else:
                                    logging.info(
                                        msg='No interpreter is instantiated!')
                                    continue
                            agent_return.state = agent_state
                            agent_return.response = action
                        else:
                            agent_state = (
                                AgentStatusCode.PLUGIN_START if name
                                == 'plugin' else AgentStatusCode.CODING)
                            if agent_state != last_agent_state:
                                # agent_return.state = agent_state
                                agent_return.response = language
                                yield deepcopy(agent_return)
                            agent_return.state = agent_state
                            agent_return.response = action
                    else:
                        agent_state = AgentStatusCode.STREAM_ING
                        agent_return.state = agent_state
                        agent_return.response = language
                    last_agent_state = agent_state
                    yield deepcopy(agent_return)
            if name:
                action_return: ActionReturn = executor(action['name'],
                                                       action['parameters'])
                action_return.thought = language
                agent_return.actions.append(action_return)
            inner_history.append(dict(role='language', content=language))
            if not name:
                agent_return.response = language
                break
            elif action_return.type == executor.finish_action.name:
                try:
                    response = action_return.args['text']['response']
                except Exception:
                    logging.info(msg='Unable to parse FinishAction.')
                    response = ''
                agent_return.response = response
                break
            else:
                inner_history.append(
                    dict(role='tool', content=action, name=name))
                inner_history.append(
                    self._protocol.format_response(action_return, name=name))
                agent_state += 1
                agent_return.state = agent_state
                yield agent_return
        agent_return.inner_steps = deepcopy(inner_history[offset:])
        agent_return.state = AgentStatusCode.END
        yield agent_return
