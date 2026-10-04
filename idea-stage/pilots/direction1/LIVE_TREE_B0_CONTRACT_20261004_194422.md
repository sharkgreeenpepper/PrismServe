# 方向一 B0：在线树事件/API 合同与 fixture 审计

日期：2026-10-04

## 结论

R001/R002 的零 GPU fixture 检查通过。新 driver 仅把已释放节点和当下副本队列快照传给 Router；一个节点完成后，ExpansionPolicy 才能根据模型输出提出孩子。事件账本证明每个孩子都在父节点完成之后释放，并覆盖了动态扇出、模型终止、节点预算剪枝和全树结束。

这只是 API 边界与事件顺序的 fixture 验证，不是在线推理结果，也不代表方向一的调度主张得到支持。

## 已实现边界

`idea-stage/pilots/direction1/live_tree_contract.py` 定义了这些接口：

- `ReleasedNode`：只包含 tree/node/parent ID、深度和当前 prompt；没有 children、未来树或 gold answer 字段。
- `RouteObservation`：只包含当前已释放节点和各副本已观察到的 in-flight 数量。
- `ModelAdapter.generate`：生成当前节点输出。
- `ExpansionPolicy.expand`：在生成完成后解析停止/展开动作。
- `LiveTreeDriver`：先记 `node_completed`，再作展开决定，最后才记录孩子的 `node_released`。

Fixture 从根节点动态产生两个分支，其中一个继续扩展，另一个输出终止结果；达到节点上限时，下一候选被记为预算剪枝。Router 不接收完整树计划，也无法读取 gold answer。

## 运行结果

命令：

```bash
python3 idea-stage/pilots/direction1/live_tree_contract.py \
  --output-jsonl idea-stage/pilots/direction1/LIVE_TREE_B0_EVENTS_20261004_194422.jsonl
```

结果：`B0_FIXTURE_PASS`；4 个节点均完成，事件包含 4 次释放、4 次派发、4 次完成、4 次展开决定、1 次模型终止、1 次预算剪枝和 1 次全树结束。机器可读账本见 `LIVE_TREE_B0_EVENTS_20261004_194422.jsonl`。

## 限制与下一步

现有 `run_vllm_placement_pilot.py` 仍在推理前构造完整 `TreePlan`，其 Router 可遍历 `node.plan.children`；因此它不满足 B1 的隐藏未来拓扑要求。新 driver 目前只接入 fixture adapter，没有 vLLM adapter、冻结的模型输出 expansion 格式或预测器。B1 不能启动。

下一步先把现有 vLLM 请求层接到 `ModelAdapter`，冻结一个在线展开/终止策略，并证明真实模型响应只在 parent completion 后产生孩子；随后再对账 GPU 额度和新问题 cohort。若模型输出几乎不产生可变分支，或预测项不改变派发，按实验计划停止方向一的方法主张。
