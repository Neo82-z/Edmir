#!/usr/bin/env python3
"""Build a local evidence report from computed metrics; no model execution."""
from pathlib import Path
from collections import Counter,defaultdict
import json,hashlib,statistics as st
H=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
b=read(H/'baseline-analysis.json');r=read(H/'route-analysis.json');t=read(H/'timeline-analysis.json');integ=read(H/'integrity.json')
lines=[]
def add(s=''):lines.append(s)
add('真实 MoE 与算子实验的证据对照分析（2026-11-17）')
add('范围：仅分析已回收数据。未连接服务器、未租机、未训练或执行预测器、未改变六个冻结形状、未修改论文源码。')
add('结论：真实模型的运行基线和路由/执行几何证据已经补上；EDM-IR 在真实 MoE 中的性能收益、冻结预测器的应用选择质量、通信隐藏的因果证据尚未补上。这轮是应用边界验证，不是算法接入后的 A/B 验证。')
add('\n一、数据验收和统计单位')
add(f"重新核验 {integ['recovery']['files']} 个原始文件、{integ['recovery']['bytes']} 字节，全部与回收清单匹配；{integ['protection']['checked']} 个保护文件的 SHA256 不变。")
add('基线：12 个正式测试块，4 种条件各 3 次，共 960 请求，122880 输出 token，全部成功。8 个短功能请求仅验证非空生成，不证明语义质量。')
add('路由诊断：320 个正式请求成功；25 个选中步骤，8 rank、46 MoE 层，9200 条真实路由记录、18400 条 gate/up 与 down 物理 kernel 记录。194246 个 token 观测 ×46层×top4 =35741264 次专家分配。这个数字不是独立请求数，也不是独立实验样本数。')
add('时间线：两个 SQLite 片段，共有 2375291 条已记录 kernel 事件。四条件完整时间线未完成；不能用文件大小或 kernel 数量替代采集质量验收。')
add('每块先按请求算 p50/p95，再取三块对应分位数的中位数；TPS 取三个整块吞吐的中位数。三次重复不足以建立稳定的分布置信区间，不把 960 个请求当作独立运行重复。')
add('\n二、原生 vLLM 基线说明了什么')
add('8×A100 SXM4 80GB，NV12，全互联；Verda 芬兰二区；driver 580.178.04，vLLM 0.20.1，PyTorch 2.11.0+cu130。GLM-4.7-Flash，BF16，固定 revision。TP1/DP8/EP8；64 routed experts、top4，每 rank 8 专家，46 个 MoE 层。')
add('Triton MoE + allgather_reducescatter，TRITON_MLA attention；prefix cache/DBO/EPLB/speculative 关闭。性能路径保留默认编译、CUDA graph 和异步调度；路由路径独立使用 eager/synchronous 并加记录器，二者延迟不可直接对比。')
add('请求是固定的受控合成中文/英文/代码输入，共128个源样本；输入长度512或4096、输出固定128、ignore_eos=True、temperature=0。不是线上真实用户流量。计时使用 native AsyncLLM streaming，包含引擎排队，不含 HTTP 和 tokenizer，也不含模型加载/编译预热。')
add('条件 输入/并发 | TPS中位数[三轮范围] | TTFT p50/p95(ms) | TPOT p50(ms) | 请求时延p95(ms)')
for w,ip in [('S1','512/1'),('S32','512/32'),('L1','4096/1'),('L32','4096/32')]:
 x=b['workloads'][w];z=x['tps'];add(f"{w} {ip} | {z['median']:.2f} [{z['min']:.2f}, {z['max']:.2f}] | {x['ttft_seconds_ms']['median_of_block_p50']:.2f}/{x['ttft_seconds_ms']['median_of_block_p95']:.2f} | {x['tpot_seconds_ms']['median_of_block_p50']:.2f} | {x['wall_seconds_ms']['median_of_block_p95']:.2f}")
add('TTFT/TPOT列为“三轮各自请求p50的中位数”，p95列同理；不是池化分位数。TPS包含每块并发填充和排空，不是饱和吞吐。并发32是全局请求并发，不是每GPU32。')
add(f"并发1→32时，短输入吞吐增为 {b['workloads']['S32']['tps']['median']/b['workloads']['S1']['tps']['median']:.2f} 倍，长输入为 {b['workloads']['L32']['tps']['median']/b['workloads']['L1']['tps']['median']:.2f} 倍。这是同一八卡系统上的负载变化，不是八卡相对单卡的扩展效率。")
add('长输入主要抬高首 token 等待；单并发 TPOT 接近，但并发32的 TPOT也变长。这支持分别呈现 prefill/decode 和不同并发条件，不能只给一个 tokens/s。尚不能从此分解出各算子或通信的时间贡献。')
add('L32 TPS极差/中位数=14.94%；S32 TTFT p50极差/中位数=40.15%。全部轮次保留。5秒间隔遥测下，L32的SM时钟均为1410MHz，但功耗/利用率有差异；不能据此证明没有短时波动，也不能唯一归因于设备或调度。')
add('新增复核：同一输入哈希、source_id与条件，在三轮中128个输出token完全相同的请求数：S1 32/32、L1 32/32、S32 0/128、L32 30/128。说明高并发路径没有给出逐token的重复运行确定性保证；不能直接据此判定答案错误、数值不合格或预测器失效。以后若声称替换算子不影响结果，仍需独立数值/质量比较，不能以请求success代替。')
add('\n三、与原算子实验的对应关系')
add('原四点 (P,M,K,N)：grouped=(8,64,8192,1024)，M512=(8,512,8192,512)，large-N=(8,64,4096,2048)，M256=(8,256,8192,512)。原算子输入/输出FP16。')
add('真实投影：gate/up的(K,N)=(2048,3072)，down=(1536,2048)，输入/权重/输出BF16，Triton累加FP32。两个真实投影均未匹配原四点的(K,N)。旧历史P8开发扫描包含K=2048，但没有N=3072或K=1536；因此也没有这两个完整投影组合。仅N=2048或K=2048相同不构成完整形状匹配。')
add('最严格的直接对应数量：原四诊断点中，0/4能作为这轮真实投影的同条件测量；真实两类投影中，0/2已有原实验的同dtype、同通信、同kernel身份候选标签。这不是“研究毫无关联”，而是不能复用具体加速比和候选排序。')
add('可以对应：A100 SM80、8 rank、NVLink/NV12；分块GEMM与通信的资源约束问题；小M、尾块、并行供给、数据移动和归并精度这些机制问题。不能等同：供应商/设备UUID、驱动、CUDA、dtype、通信语义、算子融合边界、CUDA graph/调度条件。')
add('M的关键区别：旧M是每个源s送往目的d的均匀块 X[s,d] 的行数，同一目的权重W[d]作用于P个源块，合并host的行数为P×M。真实MoE应至少记 M[s,e]（来源s送往专家e的行数）以及 M[e]=sum_s M[s,e]（专家汇总行数）。8GPU不是8专家；本模型有64个不同权重的路由专家。不能直接令旧M等于batch size、单GPU token数或M[e]。')
add('旧四点目的端总行数P×M为512、4096、512、2048。真实样本既有decode的M[e]=1，也有长prefill的M[e]>14000；旧点只触及部分行数规模，且K/N未匹配。不能仅按“相近M”给出应用覆盖率。')
add('通信也不同：旧host为均匀NCCL send/recv后合并GEMM，direct从peer LSA读取源块；真实路径先在来源做top-k，dispatch all_gatherv记录hidden states、top-k weights、top-k IDs，然后本地多专家grouped GEMM，combine用reduce_scatterv。源码接口名字里叫all2all_backend，不等于本轮实际做了原论文的pairwise All-to-All。')
add('旧q是split-K，旧g是CTA在N方向覆盖宽度。18400条实际kernel记录的SPLIT_K全部为1；vLLM的GROUP_SIZE_M=1或16是另一种block遍历分组参数，不能当成旧grouped-N的g=1/g=2。真实数据未执行旧q2/q4/q8或固定q4/g2策略。')
add('\n四、新的实测机制证据：动态M、尾块、dummy与负载不均衡')
add('下表M为汇总后的真实专家M[e]；slot/real为本地专家对齐槽位总数/真实路由行数。它包含dummy与tile padding，不是实际执行FLOPs比，更不是时间开销倍率。')
add('条件/阶段 | 非空M[e]范围 | 空专家比例 | slot/real | rank分配max/mean中位数 | 选中/全阶段步骤')
order={'S1':0,'S32':1,'L1':2,'L32':3};phase={'pure_prefill':0,'mixed':1,'pure_decode':2}
for g in sorted(r['groups'],key=lambda g:(order[g['workload']],phase[g['category']])):
 h=g['global_expert_M'];k=g['projections']['gate_up'];add(f"{g['workload']}/{g['category']} | {h['nonzero_min']}–{h['max']} | {100*h['zero_fraction']:.2f}% | {k['slot_to_real_row_ratio']:.3f} | {g['rank_assignment_max_over_mean']['median']:.3f} | {g['sampled_steps']}/{g['total_observed_steps']}")
add('gate/up与down的本地专家槽位/真实行数在这些样本中一致。这里的空专家比例按“专家×层×已选步骤”计数；rank负载按每层各rank收到的真实路由行数计算，不是GPU耗时或SM利用率。')
add('1）decode不是旧M64的自然替身。单并发每个decode步只有1个真实token，但top4分到4个专家，64专家中93.75%为空；八rank参与同步，物理输入仍保留至少8行（1真实+7dummy）。S32/L32的全局32个decode token，被top4和路由拆成不均匀专家批次；对应source-expert M[s,e]最大仅4，而汇总M[e]最大32/27。')
add('2）实际kernel主动切换tile配置：decode为 BM16/BN64/BK128、4warps/4stages；prefill/mixed为 BM128/BN128/BK64、8warps/3stages。两者SPLIT_K都为1。K/N均整除观测tile，主要可见尾块在专家M方向。这个实现已经针对小批和大批分开配置，不能把旧固定tile外推结果视为它的表现。')
add('3）单并发decode槽位膨胀很大。S1选中样本552个真实专家行对应23984槽位，L1的552行对应47696槽位；其中无真实行专家的槽位分别15152和38864。真实行自身按BM16最小对齐都只需8832槽位。增加部分来自dummy触及的专家及其块对齐。由于mask、专家跳过、不同阶段成本等因素，43.45/86.41倍槽位绝不能写成43/86倍计算量或加速潜力。S1/L1原生TPOT很接近，也说明单一槽位计数不能预测整模型时延。诊断与基线执行条件不同，二者不能逐步骤配对。')
add('4）prefill的情况相反：L32纯prefill与mixed对齐槽位只比真实行数多约3.2%/3.1%，但专家负载很不均匀；纯prefill非空M最高14298，mixed最高14425。rank最大分配量/均值的中位数约1.40/1.37。S32/L32 decode相应约2.25/1.88。这是行数不均衡证据，尚不是关键路径受最忙rank支配的因果证明。')
add('5）在L32选中纯prefill步骤，约27.75%的真实专家分配行落在M[e]>4096的专家上；mixed约27.05%。旧四点目的端最多4096行，因此即使只看行数，旧点也不能完整代表该负载；这些占比只针对选中步骤的行数，不是生产频率或执行时间覆盖率。')
add('6）读取实际top-k归属后，选中各阶段每个真实token涉及的不同目的rank均值为3.21–3.38，远端目的均值2.79–2.96。当前AllGather语义仍向所有rank分发，而不是只送这些目的rank。此结果描述通信语义差异；不能把7/2.9直接当作流量加速比，因为实际collective算法、dummy、packing与combine都不同。记录器的logical_tensor_bytes也不能当成NVLink实测流量。')
add('上述动态M、padding和路由归属是新观察，不是冻结预测器“已经理解了这些机制”的验证。本轮既未给预测器喂新标签，也未重新评分/训练。')
add('\n五、采样代表性：哪些覆盖完整，哪些不完整')
add('选中25步中，8rank×46层的路由守恒和两投影kernel几何完整，检查无失败；这一层面可以称为完整。每阶段只取最先3步，不能称为所有推理步骤或线上请求的代表性抽样。')
add('完整诊断计数：S1/L1各32个纯prefill、4064个纯decode；S32为4个纯prefill、508个纯decode；L32为1个纯prefill、22个mixed、499个纯decode。全阶段计数来自独立诊断执行，不能替代原生异步基线的阶段频率。')
add('S1/L1只有3/4064=0.074%的decode步骤含完整路由细节；L32为3/499=0.601%。若把采样到的三步prefill和三步decode等权平均，会严重改变真实步骤比例；同样不能把被选中的大prefill按其token数加权后宣称代表端到端耗时。保留阶段分层表，不报全应用单一M覆盖率。')
add('受控输入包含重复/拼接生成文本，可能影响专家偏好。仅一个模型、一个硬件互联条件和两种输入长度/并发，尚不足以外推到其他MoE、生产对话或不同拓扑。')
add('\n六、时间线失败的进一步定位')
add('第一片段对应成功的diag-S1请求，第二片段在后续采集中遭遇GPU1 Xid69，继而CUDA unspecified launch failure / CUBLAS_STATUS_INTERNAL_ERROR。不能把第二片段当作完整正常运行。')
add('这次SQL复核发现，第一片段的DIAGNOSTIC_EVENT明确记录：当前CUDA driver version 13.0不受该Nsight Systems版本支持，改用driver version 12.9的追踪库。此告警覆盖全部8个worker。两个片段的8个worker均有“Not all CUDA events might have been collected”，并有NVTX可能不完整的提示。')
add('因此需要区分三个层次：①可以确认观测到AllGather/ReduceScatter和fused_moe_kernel；②片段中有8卡事件，不等于事件完整；③版本不匹配提供了明确的采集风险线索，但还不能证明它导致Xid69。是否为驱动/工具兼容性、设备或执行路径故障，现有记录无法唯一归因。')
add('本报告不计算/发表overlap百分比、通信隐藏比例或由kernel duration求出的服务瓶颈占比。NCCL kernel持续时间还可能包含同步等待；跨8卡duration求和不是墙钟时延。无profiler基线和eager路由诊断均在故障前独立完成，不因后续追踪失败全部作废。')
add('可以写入论文的失败说明（一句话）：原生性能与路由采集完成；细粒度追踪因工具兼容性告警及设备执行异常未完成，故不据此报告通信隐藏比例。')
add('\n七、论文论点逐项判定')
claims=[
('计数恒等式与候选资源权衡','算子范围已支持','旧固定执行中CTA数、请求字节、split-K反馈读写计数可推导并对实现核对；必须保留模型假设。','不能把请求字节当成物理链路流量或完整延迟下界。'),
('精度约束先于候选选择','FP16算子范围已支持','旧FP16中间归并的失败被保留并排除；新增FP32 partial路径48/48检查通过且有额外代价。','BF16应用里的EDM-IR替换尚未实现/验收；原生BF16成功不验证它。'),
('直接访问/融合候选在部分形状优于强host','原算子范围已支持','四诊断点最佳已测候选相对合并host约1.69–3.16倍；最终large-N复测约1.88倍。','这是候选oracle收益，不是预测器收益或真实MoE收益。'),
('流水调度可减少完整程序时间','局部配对证据成立','同条件pipeline/serial：M512降低10.33–16.36%，large-N降低7.09–9.99%。','pipeline仍慢于该阶段合并host/direct；没有测出真实MoE隐藏了多少通信。'),
('加入耦合特征改善选择损失','开发诊断有限支持','四形状留一：均值regret12.60%→1.22%，最大47.36%→3.05%；首选仍仅2/4。','固定q4/g2平均1.04%、最大2.92%更好；不能声称模型优于固定策略。'),
('冻结预测器能迁移、双向选择与正确排序','未闭环且存在反证','最终large-N选q8/g2，汇总损失3.48%、最坏块8.42%；六对fused内部汇总方向均反。','新增真实MoE没有候选计时标签，不能当作预测成功。六个冻结形状未验证。'),
('真实MoE确有动态专家行数和执行padding','本次新增直接证据','25步×8rank×46层，路由守恒、归属、BF16物理几何、tile配置和padding核验通过。','仅证明选中诊断样本，不代表生产频率/全模型时间份额。'),
('八卡真实模型原生运行与基线','本次新增直接证据','960请求全成功，四条件三重复的吞吐/TTFT/TPOT及波动明确。','只对固定native配置成立，不是最大服务能力、质量评估或优化后性能。'),
('原四形状代表实际模型的大部分成本','未闭环','真实两投影K/N均不同，动态M和通信实现也不同。','没有相同执行边界的投影计时和阶段占比，不能声称覆盖多少模型时间。'),
('EDM-IR接入vLLM产生端到端收益','未闭环','本次运行的是原生vLLM，没有EDM-IR开/关A/B。','尚缺语义匹配的接入、BF16数值验收、同请求同环境A/B、必要的质量比较。'),
('共享资源争用解释性能、通信被计算隐藏','未闭环','旧算子有交叉效应，本轮有实际collective及专家几何观察。','缺可靠的完整时间线、同条件隔离对照和物理流量/资源证据；不能用叠加图代替因果验证。')]
for i,(claim,status,yes,no) in enumerate(claims,1):add(f'{i}. {claim}\n   判定：{status}\n   已有：{yes}\n   边界：{no}')
(H/'claim-matrix.json').write_text(json.dumps([dict(claim=c,status=s,evidence=e,limit=l) for c,s,e,l in claims],ensure_ascii=False,indent=2)+'\n')
add('\n八、对预测器失败的判断保持不变')
add('当前论文采用共享的正则化log延迟回归评分，不是正在训练的深度分类器。既有审计显示封存解满足正规方程、候选重排不影响评分；没有证据说明改优化器就能消除错误。类别大方向正确与fused内部排序失误同时存在；现有四点没有同身份host获胜样本，不能证明双向分类能力。')
add('这轮提供的是更具体的模型边界：M[s,e]/M[e]、专家权重数、路由负载、dummy、tile选择、精度和collective路径都不同。它支持“不能只靠均匀M和孤立算子代价迁移到完整系统”的解释；不支持把原large-N误选的根因直接归咎于这些新增因素，因为原large-N本来没有真实路由/dummy。')
add('冻结模型上既无本轮候选预测，也无host/q2/q4/q4g2同条件真值，无法计算真实MoE regret、分类准确率或排序准确率。不开新训练，不为已有失败新增事后修正项。')
add('\n九、论文应如何收束')
add('可以采用两层证据结构：主体保持“受数值与执行条件约束的通信-GEMM候选选择”；真实MoE作为应用案例，展示实际路由如何产生不同的M、padding与通信边界，并给出原生vLLM运行基线。先在背景/实验开头明确代理算子和真实执行的区别，再呈现两个层次的结果。')
add('建议新增：一张原生基线表、一张阶段化专家M/空专家/padding表、一张算子与应用条件对照表。不要把operator speedup画成vLLM speedup。论文旧限制中“没有真实路由轨迹”已经过时，可改为“具有受控工作负载的路由轨迹与原生基线，但未完成EDM-IR接入后的端到端比较”。')
add('固定q4/g2仍放在算子选择基线中；本轮原生Triton不能伪装成它。明确固定策略是事后选定、四点是开发诊断，六个保留形状未评测。失败只需一两句说明，并保留完整附录/回执。')
add('若论文主张止于算子选择及应用边界，本地材料足以整理这部分结果，无需继续修改预测器。若主张升级为真实MoE推理加速，则还欠接入与同条件A/B，这是新增证据类型，不能靠分析现有日志补成。本次不启动该实验。')
add('\n十、建议可直接采用的应用实验表述')
add('我们在8张A100 SXM4 80GB上运行BF16 GLM-4.7-Flash，并采用固定的短/长输入与单/多并发受控请求集。原生vLLM基线完成960个请求；独立路由诊断覆盖25个选中步骤、全部8个rank及46个MoE层。单并发decode的非空专家每步仅接收一个真实token，而长输入prefill的专家行数可超过14000，并具有显著的负载和padding差异。这些观测说明均匀通信-GEMM代理的形状与执行条件不能直接替代应用行为。本实验尚未将EDM-IR候选接入vLLM，因此不报告预测器的应用选择收益或端到端加速。')
add('\n复现与文件')
add('在 /Users/neo/code/llm-serving 下运行：')
add('python3 experiments/2026-11-17-real-moe-analysis/analyze.py')
add('python3 experiments/2026-11-17-real-moe-analysis/analyze_timeline.py')
add('python3 experiments/2026-11-17-real-moe-analysis/build_report.py')
add('Python标准库，无网络，无CUDA，无预测器import。输入只读；输出在本目录。')
add('baseline-analysis.json：12块原始请求重算；route-analysis.json：M分布、padding、采样、rank负载、tile配置与通信接口；timeline-analysis.json：SQLite原始告警和kernel清单；claim-matrix.json：论文论点证据矩阵；integrity.json：267个文件与17个保护文件复核。')
add('旧算子依据：../2026-11-17-evidence-synthesis/REPORT.txt、results.json；../2026-11-17-final-operator/analysis.json；../2026-11-16-shadeform-precision/stage-summary.json。真实数据依据：../2026-11-17-real-moe-gpu/recovered/。所有失败保留。')
(H/'REPORT.txt').write_text('\n'.join(lines)+'\n')
print('Written REPORT.txt and claim-matrix.json')
