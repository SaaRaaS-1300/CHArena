# 特殊包
import sys

# import Xxx
from config.stakers.internlm2 import RoleplayerAgentI2
from build.personas import RoleplayerModel
from lagent.agents.internlm2_agent import Internlm2Protocol
from lagent.llms.meta_template import INTERNLM2_META

# 参数准备


# 环境准备


# 网络准备

# /root/horowag_mini/model/InternLM2-chat-1_8B
# /root/share/model_repos/internlm2-chat-7b

# 创建语言模型的实例
Horo = RoleplayerModel(
            path="/root/horowag_mini/model/InternLM2-chat-1_8B",
            top_k=1,
            top_p=0.75,
            temperature=0.75,
            max_new_tokens=128,
            repetition_penalty=1.001,
            meta_template=INTERNLM2_META,
    )

# initialize protocol
protocol = Internlm2Protocol(
    meta_prompt=('你是一个卧底，正在参与谁是卧底的游戏。你需要让别人不认为你是卧底。同时假装别人才是卧底。最终赢得胜利。'),
    interpreter_prompt="",  # 置空
    plugin_prompt="",  # 置空
)

# initialize agent
agent = RoleplayerAgentI2(
    llm=Horo,
    protocol=protocol,
    max_turn=3
)

# 使用 RoleplayerAgentI2 的 chat 方法进行聊天
history = []  # 初始化历史记录为空列表
running = True

try:
    while running:
        print("请输入:")
        # 获取用户的消息
        message = sys.stdin.readline().strip()

        # 如果用户输入了 "quit"，那么结束循环
        if message.lower() == "quit":
            running = False
            continue

        # 调用 chat 方法
        agent_return = agent.chat(message=message, history=history)

        # 打印助手的响应
        assistant_message = agent_return.response
        output = agent_return.inner_steps[-1]['content']

        print()
        print("horo的回复：", output)
        print()

        # 更新历史记录
        history = agent_return.inner_steps
        print("历史记录：", history)

except KeyboardInterrupt:
    running = False
    print("\nLoop interrupted by user (Ctrl+C was pressed)")
    