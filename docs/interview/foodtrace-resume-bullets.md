# FoodTrace 简历要点

以下表述只引用固定 seed 仿真 benchmark，不暗示线上生产指标。

- 设计并实现食品过敏原召回调查闭环，以固定 Cypher 追溯原料批次到生产批次，再用参数化只读 SQL 查询库存和订单，显式约束图 → SQL 数据依赖。
- 构建稳定 Evidence ID、指标到证据类型/payload 的绑定校验，以及 `ALLOW_REPORT / HUMAN_REVIEW / BLOCKED` 保守 Gate，阻止缺证据数字和依赖失败时生成召回报告。
- 建立 11 场景确定性安全 benchmark，覆盖图/SQL 冲突、缺图边、SQL 宕机、SOP 缺失、范围扩张和 SQL 越界；本地实测 11/11 行为匹配、冲突转人工率 100%、不安全 SQL 阻断率 100%。
- 为 CLI 与 FastAPI 提供同一 workflow 的薄入口，输出产品、批次、门店、订单集合和七段结构化 trace；本地全量回归 143 项通过。

面试时同时说明限制：上述结果来自固定 seed 仿真，不是线上召回率；本机 Docker daemon 未启动时，只能证明 fixture 闭环，不能宣称真实 Neo4j/MySQL 已完成运行验证。
