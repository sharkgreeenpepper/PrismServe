## PrismServe（棱镜推理）

正式中文名：

> 面向多 GPU 资源协同的大模型推理优化研究

英文全称：

> PrismServe: Resource-Aware Co-Optimization for Multi-GPU LLM Inference

“Prism”表示从多个维度分析推理系统：

- 计算资源；
- 显存与 KV Cache；
- 多卡通信；
- Prefill/Decode 调度；
- SLO 与吞吐；
- MoE 或 Speculative Decoding。

这样以后即使最终主线从多卡通信转向 KV Cache 或调度，项目名也不需要修改。

例如：

- 通信方向：PrismServe 的拓扑感知服务路径选择；
- KV 方向：PrismServe 的 KV Cache 迁移与重计算决策；
- 调度方向：PrismServe 的 Prefill/Decode 协同调度；
- MoE 方向：PrismServe 的层次化专家通信优化。

其他可选名称：

1. **InferWeave（织推）**  
   面向大模型推理的计算、存储与调度协同优化。

2. **FlexServe（灵推）**  
   面向多 GPU 集群的自适应大模型推理系统。

3. **ModelPulse（模脉）**  
   面向动态负载的大模型推理资源调度与性能优化。

4. **MosaicInfer（拼推）**  
   面向异构资源的大模型推理协同优化。

5. **OmniInfer（全域推理）**  
   面向计算、存储、通信和服务质量的统一推理优化框架。

如果希望名字更适合学位论文，我建议最终采用：

> **PrismServe：面向多 GPU 资源协同的大模型推理优化研究**

其中 `PrismServe` 作为项目/系统名，后面的具体研究方向作为论文副标题。正式使用前再检索一下 GitHub、Google Scholar 和论文数据库，避免与已有项目重名。
