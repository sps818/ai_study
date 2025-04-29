### Agent的本质：
- 大模型可以借助外部工具来实现一些特定的功能
- 流程：
    - 准备外部工具（函数）列表 tools
    - 准备大模型 model
    - 创建一个 agent
    - 调用 agent

### Agent 的手动创建：
- 借助 LangGraph 搭建工作流
- Graph:
    
    - 节点 Node：
      - 执行一个动作：
        - 调用函数？ -- tool 节点
        - 调用模型？-- model 节点
        - ...
    
    - 边 Edge:
      - 节点之间的连线
      - 代表着状态转移
      - 固定边
      - 条件边

    - 状态 State:
      - 最常用的就是 Message State
      - 状态中存储着消息


### Agent 的本质：

- model 一个大模型
  - 做决策大脑
- tools 一些列工具
  - 做能力拓展（专长）
- create_react_agent 创建一个 agent
