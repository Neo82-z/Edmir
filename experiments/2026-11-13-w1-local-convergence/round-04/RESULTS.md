# W1 本地收敛结果

只使用四个开发形状；六个冻结形状未读取。旧数据、失败记录与旧模型均保留。

52 个主候选组合中，42 个通过全部三个输入种子；10 个失败。4 个 pipeline 对照属于单独阶段，不混入评分。

下表是相同固定 ridge 校准的整形留出诊断。original_geometry 不是旧模型复现；coupled_geometry 只新增可解释计数。未按验证损失调整系数、正则或候选集。

|表达|形状|选择|实测最快|选择损失|最坏重复损失|方向错误|
|---|---|---|---|---:|---:|---:|
|original_geometry|grouped|splitk8_groupedn2|splitk8_groupedn2|0.00%|0.90%|9/55|
|original_geometry|m512|splitk4_groupedn2|splitk4_groupedn2|0.00%|0.00%|3/36|
|original_geometry|large_n|splitk8_groupedn2|splitk2|5.26%|5.83%|14/55|
|original_geometry|m256|splitk8_groupedn2|splitk8_groupedn2|0.00%|0.00%|3/55|
|coupled_geometry|grouped|splitk8_groupedn2|splitk8_groupedn2|0.00%|0.90%|9/55|
|coupled_geometry|m512|splitk4_groupedn2|splitk4_groupedn2|0.00%|0.00%|2/36|
|coupled_geometry|large_n|splitk8_groupedn2|splitk2|5.26%|5.83%|14/55|
|coupled_geometry|m256|splitk8_groupedn2|splitk8_groupedn2|0.00%|0.00%|5/55|

每个候选的分数、错误方向、三轮差值、训练形状和系数保存在对应 JSON；所有失败都保留，不按波动删样本。

FP32 跨片存储有独立实现与候选身份，16 个组合待 GPU 验证，没有借用原 FP16 的计时或通过状态。

四个开发形状都以 fused 为优；现有数据无法证明双向 host/fused 泛化。新增计数不是实测占用率，完整端到端耗时也不能分解出通信与计算争用的因果贡献。
