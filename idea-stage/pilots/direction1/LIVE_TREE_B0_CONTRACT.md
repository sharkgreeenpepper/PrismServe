# 方向一 B0：在线树接口、路由反事实与失败账本审计

日期：2026-10-04

## 结论

零 GPU 合约、失败路径和合成路由夹具通过。路由器只接收当前已释放节点和同一时刻的副本队列快照；孩子由模型完成父节点后才提出并释放。proposed 路由还会在完全相同的已释放批次、队列快照和父节点放置历史上计算一份关闭后代预测信号的 QKV 反事实分配。因此 B1 可以用这份逐批次反事实判断预测项是否直接改变分组，不再把两次运行之间的队列差异误算成机制效果。

这只证明代码路径和审计口径成立，不是模型预测精度、真实在线表现或性能收益证据。

## 已实现边界

- `ReleasedNode` 和 `RouteObservation` 不带未来 children、预生成树或 gold answer。
- `LiveTreeDriver` 先记录节点完成，再解析其展开/停止输出，最后释放孩子。
- `QueueKvRouter.choose_batch` 对同一 ready frontier 联合优化副本分组；proposed 会同时记录同快照 QKV assignment。
- 校准文件绑定源运行配置、数据集 ID、数据集 SHA-256 和实际事件索引；test runner 拒绝数据哈希不一致及校准/测试索引重叠。
- 失败请求会记为 `node_failed`，兄弟取消、派发前中止和树失败也分别入账。
- 树延迟分位数只纳入成功完成树；失败树单独保留 `terminal_elapsed_s`。事件与 summary 使用一致的 released/completed/failed/cancelled/aborted 计数口径。

## 零 GPU 结果

运行命令：

```bash
python3 idea-stage/pilots/direction1/live_tree_contract.py \
  --output-jsonl idea-stage/pilots/direction1/LIVE_TREE_B0_EVENTS_20261004_203129.jsonl \
  --failure-output-jsonl idea-stage/pilots/direction1/LIVE_TREE_B0_FAILURE_EVENTS_20261004_203129.jsonl
python3 idea-stage/pilots/direction1/audit_live_tree_routing.py \
  --output idea-stage/pilots/direction1/LIVE_TREE_B0_ROUTING_20261004_203129.json
```

- 正常树：4 个节点释放、派发、完成和动态展开；覆盖 1 次模型停止、1 次节点预算剪枝，最终树完成。
- 失败树：3 个节点释放/派发；记录 1 个失败、1 个兄弟取消、1 个已完成节点和整树失败。
- 合成 ready batch：QKV assignment 为 `[0, 0, 1]`；关闭预测项的同快照反事实也是 `[0, 0, 1]`；加入合成后代估计后 proposed assignment 为 `[1, 0, 1]`。

输出中的投影负载数值只用于确认信号能改变 assignment，不可解释为实测服务时间。

## B1 启动前冻结

- Cohort 文件保存在机器本地 infra cache，未放入 Git。
- 数据集：`gsm8k:test`，SHA-256 `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`。
- 校准索引：`18, 56, 300, 340`；配对测试索引：`420, 743, 976, 996, 1049, 1058, 1166, 1210`。它们与扫描到的 44 个已用/预留索引均不重叠，校准集与测试集也不重叠。
- 父预算账本：`direction1-gsm8k-independent-20261003-seed20261003`，400 GPU-min 上限、已用 142.655、账面剩余 257.345 GPU-min、保留清理预算 20 GPU-min。B1 单独硬上限为 80 GPU-min，使用 GPU 4–7；没有借用其他 run ledger。
- 复用已有 vLLM 0.30.0 + CUDA 12.9 + 70B 模型；双 TP=2 副本计划监听本机 8004/8005。启动前 GPU 空闲、端口未监听，但模型服务尚未启动。
- 服务率、队列时间和 prefix cache 比例使用同模型历史测量形成 B1 输入；这只是路由估计参数。B1 仍以真实返回的 request metrics 和 cached prompt tokens 检查遥测是否完整。

## 限制

B1 只是机制筛查，8 个测试问题不支持稳定 p95 或论文级性能主张。即使路由子门通过，也必须同时满足两臂 8/8 成功完成且必需 request/cache 遥测齐全，才算 B1 screen 通过。B1 不与 TOPAS/FATE、Preble 或 Autellix 作性能比较；只有机制门通过，才评估是否值得进入有限强基线筛查。
