# 特殊包
import sys

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
    