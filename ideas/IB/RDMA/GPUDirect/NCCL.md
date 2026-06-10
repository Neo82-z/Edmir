通信拓扑层面很关键，但是我没有办法单纯从拓扑推出llm-serving的效益，但是能从根本上理清整个推理回环。
## https://zhuanlan.zhihu.com/p/2046220669998851708  一篇理清主线的文章 infra全过程

首先要把网络 通信 推理这三块分开来看，虽然整体来看是一个系统，但是本质上网络和通信都是在为推理做铺垫
概念上：
LLM workload：
  -> vLLM / PyTorch distributed / Megatron
  -> NCCL collective 或 P2P(Mooncake发过一篇文章详细做了p2p:https://www.usenix.org/conference/fast25/presentation/qin)
  -> NCCL NET/IB backend
  -> GPUDirect RDMA
  -> InfiniBand HCA / switch / fabric
  -> remote GPU memory


1.InfiniBand(还挺重要，自己目前也不是非常了解)：一种高性能网络 fabric，包含 HCA 网卡、IB switch、subnet manager、链路层、传输层。 IB详细可见：https://network.nvidia.com/pdf/whitepapers/Intro_to_IB_for_End_Users.pdf
2.RDMA：能力，不是某一种线。意思是远端直接内存访问，减少 CPU 参与和内存拷贝。 NV在这块的文档很多：https://docs.nvidia.com/doca/archive/doca-v2.2.0/pdf/rdma-programming-guide.pdf
3.GPUDirect RDMA：NIC 可以直接读写 GPU 显存，这对 NCCL 多节点性能非常关键。
4.RoCE：把 RDMA 跑在 Ethernet 上，通常需要 PFC/ECN/QoS 配好，否则容易抖。同样Nv在这块文档很多但是太过工程导向：https://docs.nvidia.com/networking/display/winofv55054000/rdma+over+converged+ethernet+(roce) 关于RoCE知乎上有人说他完成了不错的工程进展（ “我们最近治理了机房roce网络、升级了框架软件驱动（这放大厂得磨蹭个大半年）， 对齐了deepep通信、 nccl通信、 te通信（8卡rdma带宽390G）， 核心算子mfu，且终于稳定上线了大ep+centric kvcache”）这个思路其实就是我想实践并研究的。
5.NCCL NET/IB：NCCL 的网络传输路径，名字叫 IB，但很多时候也能覆盖 RoCE，因为底层走 verbs/rdma 这一套。
6.NVLink/NVSwitch：单节点 GPU-GPU 互联，不是跨节点网络。它解决的是节点内（也就是我们说的卡间通信，NCCL发挥作用时刻，具体有无NVLink看的是能拿到什么样的卡）。如果是4090/5090，我们只能测 PCIe-only 路径，因此它们更适合做 NCCL/PyTorch distributed 和 profiling 流程热身；DeepEP V2 的主结论需要放到 H100/H800/H20 这类 Hopper/SM90 平台上。InfiniBand/RoCE 解决的是节点间。


InfiniBand貌似就是一个使用交换机 交换机上使用的流控方式。接收方提前告诉发送方自己有多少 buffer 的信用额度，发送方不能发送超出自己被分配的信用额度的数据。好处就是能事先预防，坏处就是贵。 https://zhuanlan.zhihu.com/p/2046220669998851708

因为IB太贵，所以RoCE想复用以太网的基础设施来跑。以太网的 VLAN PCP 有 3 个 bit 对应 8 个优先级，PFC 可以在要炸了的时候把某个优先级上的流量全给停了。PFC 的问题是总归就 3 个 bit 八个优先级，RDMA 流量分到一个优先级上，一挂就全停了。而且由于像 clos 这种拓扑会成环，可能导致永久死锁只能重启。

### 网络层面东西太多 还是有点偏离主线 后续再逐层研究 但是有几个值得讨论的问题

1.NCCL 到底走了 NET/IB 还是 NET/Socket？
2.是 GPUDirect RDMA，还是绕 CPU host memory？
3.是单 rail 还是 multi-rail？
4.all-to-all 为什么比 all-reduce 更容易炸？
5.小包 latency、大包 bandwidth、congestion、NUMA、PCIe root complex 分别在哪里影响？
6.vLLM/MoE/KV cache transfer 的通信形态到底适不适合标准 NCCL collective？

网络层面可以再IB的HCA QP CQ verbs subnet manger DMA read/write/send/recv 过渡到 通信层的NCCL_DEBUG=INFO
NCCL NET/IB
nvidia-peermem
ibv_devinfo
nccl-tests
all_reduce / all_gather / reduce_scatter / all_to_all

回到通信侧过度到推理层 Deepep是很好的开源范式。

1. vLLM 为什么需要 scaling
2. vLLM 的并行策略
   - DP: 多副本，请求级扩展
   - TP: 单个请求跨 GPU，通信最敏感
   - PP: 层间切分，有 pipeline bubble 和 activation transfer
   - EP/MoE: expert 并行，all-to-all 压力大
3. 单节点通信路径
   - NVLink / NVSwitch / PCIe
   - NCCL intra-node
4. 多节点通信路径
   - NCCL NET/IB
   - InfiniBand / RoCE
   - GPUDirect RDMA
5. vLLM 配置要点
   - Ray cluster
   - --tensor-parallel-size
   - --pipeline-parallel-size
   - --privileged
   - NCCL_IB_HCA=mlx5
   - --ipc=host / --shm-size / /dev/shm
6.  RDMA调试
   - nvidia-smi topo -m
   - ibv_devinfo
   - /dev/infiniband
   - nvidia-peermem
   - NCCL_DEBUG=INFO/TRACE
7. 与研究方向的关系
   - 通信瓶颈在哪里 (本质来说很大概率是网络)
   - 哪些 collective 被触发 (重要)
   - 是否能做 topology-aware serving
   - 是否能改调度/并行策略来避开通信热点 (很多infra工程博客已经做出示范)

DP 扩容更像 serving system 问题，TP/EP 扩容更像 communication system，所以回环可以是并行方式 -> 触发哪些通信算子 -> 消息大小分布 -> 延迟/带宽占比 -> 拓扑敏感性 -> 是否可以优化

可以问自己的一些问题：TP 的 all-reduce / all-gather 在 decode 阶段是不是小消息高频？
MoE 的 all-to-all 在 expert routing 下是不是更容易造成网络拥塞？
PP 跨节点时，是不是 activation transfer + pipeline bubble 更影响 TTFT？
KV cache transfer 如果跨节点，RDMA 能不能显著降低 P99 latency？
NCCL 走 NET/IB 和退化到 NET/Socket，vLLM benchmark 差多少？  但是这些问题分散的比较广，可以适当的收敛一些

关于最终讨论和论证的点落到哪里的话
网络和通信拓扑可以分层为Topology
  -> communication primitive cost
  -> serving workload communication pattern
  -> exposed communication on critical path
  -> TTFT / ITL / throughput / P99(延迟)      然后去证明在什么 workload、什么并行策略、什么拓扑下，通信成为 LLM serving 的瓶颈；以及 topology-aware 或 communication-aware 的策略能不能减少暴露在 critical path 上的通信时间。
比如per token延迟 = compute time + exposed communication time + schcduling / memory (Kvcache) / quene overhead 还有考虑compute and communication的overlap.如果通信完全被 GEMM overlap 掉，那 NCCL 再快一点，serving 指标也可能没变化。反过来，如果 decode 阶段小 batch、高频 all-reduce 暴露在 critical path 上，那通信变动会直接影响 ITL/P99。

所以我想分三层论证：

1:拓扑本身是否改变通信成本。

NVLink/NVSwitch vs PCIe
intra-node vs inter-node
IB/RoCE/RDMA vs Socket
near NUMA pair vs cross NUMA pair
用 nccl-tests、NCCL_DEBUG、nvidia-smi topo -m 证明。

第二层：vLLM workload 到底触发了什么通信。

TP -> all-reduce / all-gather
PP -> activation transfer
EP/MoE -> all-to-all
KV transfer -> P2P / RDMA-style movement
DP -> 请求级复制，通信压力相对小
用 trace / profiler / NCCL log 证明。

第三层：这些通信是否真的影响 serving 指标。

TTFT
ITL
tokens/s
P50/P99 latency
GPU utilization
NCCL time ratio
network bandwidth utilization
用 vLLM benchmark 和 Nsight/PyTorch profiler 证明。

因为我需要先知道：1.这个 workload 的通信占比 f 是多少？
2.通信有没有被 overlap？
3.通信发生在 prefill 还是 decode？
4.消息大小是大包带宽瓶颈，还是小包 latency 瓶颈？
5，瓶颈在 GPU compute、PCIe、IB、CPU 调度，还是 KV cache memory？ 不过这都是显而易见的，已有的结论，不过是要再做一遍。



我想做一套可解释的性能模型，判断什么时候通信优化有收益，什么时候没有收益。如果只做 benchmark 堆数字，其实我自己也看不出什么

如果能知道什么时候 TP 跨节点是坏的？什么时候 PP 比 TP 更适合跨节点？什么时候 EP 的 all-to-all 会成为瓶颈？什么时候 RDMA 对 KV cache transfer 真的有用？什么时候 NCCL benchmark 很好，但 serving 指标没提升？那就还不错

但是这些貌似也是blog有所总结，我们还是要选择一些策略和方法，自动算出最优的情况。

总之这块既要保障容量 提升命中率，还要保障latency，避免阻塞计算，听说10k以上输入，用rdma带宽换tc的计算非常划算，因为已经降到秒级了，难搞的是roCE ib网卡 switch的流量qos。 还有deepseek用3fs ssd的存储冷热分级。





