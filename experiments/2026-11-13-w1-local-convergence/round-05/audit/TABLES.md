# W1 开发数据计算附表

由 analyze.py 生成。所有耗时单位为 ms；中心值为三轮进程中位数的中位数。

前三轮分别列出；波动范围 = (最大进程中位数 − 最小进程中位数) / 中心值。

## grouped — [P,M,K,N]=[8, 64, 8192, 1024]

|候选|阶段|轮 1|轮 2|轮 3|中心|波动范围|
|---|---|---:|---:|---:|---:|---:|
|splitk8_groupedn2|measure|0.595975|0.592378|0.608361|0.595975|2.68%|
|splitk4|measure|0.605039|0.607955|0.602940|0.605039|0.83%|
|splitk4_groupedn2|measure|0.622104|0.615320|0.627815|0.622104|2.01%|
|splitk8|measure|0.658914|0.656653|0.656297|0.656653|0.40%|
|splitk2|measure|0.811023|0.811244|0.814097|0.811244|0.38%|
|splitk2_groupedn2|measure|0.919153|0.919180|0.921639|0.919180|0.27%|
|host_merged_cublaslt|measure|1.166813|1.112322|1.138724|1.138724|4.79%|
|host_merged_cutlass|measure|1.242739|1.226850|1.228932|1.228932|1.29%|
|direct|measure|1.374773|1.378477|1.391593|1.378477|1.22%|
|direct_groupedn2|measure|1.653484|1.659346|1.659203|1.659203|0.35%|
|host|measure|2.343777|2.690930|2.645103|2.645103|13.12%|

独立进程最快候选：splitk8_groupedn2, splitk8_groupedn2, splitk4。

## m512 — [P,M,K,N]=[8, 512, 8192, 512]

|候选|阶段|轮 1|轮 2|轮 3|中心|波动范围|
|---|---|---:|---:|---:|---:|---:|
|splitk4_groupedn2|measure|0.934867|0.933466|0.934276|0.934276|0.15%|
|splitk2_groupedn2|measure|1.117763|1.117912|1.118581|1.117912|0.07%|
|splitk4|measure|1.549703|1.657812|1.553101|1.553101|6.96%|
|splitk2|measure|1.607143|1.581189|1.595176|1.595176|1.63%|
|direct_groupedn2|measure|1.904108|1.901968|1.900958|1.901968|0.17%|
|direct|measure|1.913660|1.916614|2.023992|1.916614|5.76%|
|host_merged_cutlass|measure|2.876246|2.876060|3.150675|2.876246|9.55%|
|host_merged_cublaslt|measure|2.899171|2.895325|2.918125|2.899171|0.79%|
|pipeline|pair|4.358424|4.302049|4.539154|4.358424|5.44%|
|pipeline_serial|pair|5.066429|5.143754|5.061939|5.066429|1.61%|
|host|measure|5.327746|5.293419|5.004712|5.293419|6.10%|

独立进程最快候选：splitk4_groupedn2, splitk4_groupedn2, splitk4_groupedn2。

## large_n — [P,M,K,N]=[8, 64, 4096, 2048]

|候选|阶段|轮 1|轮 2|轮 3|中心|波动范围|
|---|---|---:|---:|---:|---:|---:|
|splitk2|measure|0.610664|0.589496|0.586847|0.589496|4.04%|
|splitk4|measure|0.606980|0.593852|0.588642|0.593852|3.09%|
|splitk4_groupedn2|measure|0.619731|0.595406|0.605371|0.605371|4.02%|
|splitk2_groupedn2|measure|0.610826|0.605981|0.613881|0.610826|1.29%|
|splitk8_groupedn2|measure|0.620495|0.623860|0.593662|0.620495|4.87%|
|splitk8|measure|0.665946|0.665352|0.665841|0.665841|0.09%|
|direct|measure|0.796874|0.795102|0.797507|0.796874|0.30%|
|direct_groupedn2|measure|0.923776|0.921183|0.922527|0.922527|0.28%|
|host_merged_cutlass|measure|1.034550|1.026022|1.025378|1.026022|0.89%|
|host_merged_cublaslt|measure|1.122060|1.118088|1.126521|1.122060|0.75%|
|pipeline|pair|1.741907|1.707531|1.715482|1.715482|2.00%|
|host|measure|1.824331|1.808540|1.861629|1.824331|2.91%|
|pipeline_serial|pair|1.874847|1.896946|1.901577|1.896946|1.41%|

独立进程最快候选：splitk4, splitk2, splitk2。

## m256 — [P,M,K,N]=[8, 256, 8192, 512]

|候选|阶段|轮 1|轮 2|轮 3|中心|波动范围|
|---|---|---:|---:|---:|---:|---:|
|splitk8_groupedn2|measure|0.704216|0.701701|0.702721|0.702721|0.36%|
|splitk4_groupedn2|measure|0.711766|0.711066|0.710151|0.711066|0.23%|
|splitk4|measure|0.903720|0.902565|0.902366|0.902565|0.15%|
|splitk8|measure|0.985221|0.985570|0.984196|0.985221|0.14%|
|splitk2|measure|0.995537|0.995656|0.994161|0.995537|0.15%|
|splitk2_groupedn2|measure|1.068496|1.067941|1.066372|1.067941|0.20%|
|direct|measure|1.666197|1.665587|1.664817|1.665587|0.08%|
|host_merged_cutlass|measure|2.185363|1.859177|1.868752|1.868752|17.45%|
|direct_groupedn2|measure|1.879582|1.874277|1.878107|1.878107|0.28%|
|host_merged_cublaslt|measure|1.845978|2.175077|2.198377|2.175077|16.20%|
|host|measure|3.655924|3.708264|3.710178|3.708264|1.46%|

独立进程最快候选：splitk8_groupedn2, splitk8_groupedn2, splitk8_groupedn2。

## 同进程配对：正值表示前者缩短耗时

|形状|比较|轮 1|轮 2|轮 3|
|---|---|---:|---:|---:|
|grouped|direct_groupedn2 vs direct|-20.27%|-20.38%|-19.23%|
|grouped|splitk2_groupedn2 vs splitk2|-13.33%|-13.30%|-13.21%|
|grouped|splitk4_groupedn2 vs splitk4|-2.82%|-1.21%|-4.13%|
|grouped|splitk8_groupedn2 vs splitk8|+9.55%|+9.79%|+7.30%|
|m512|direct_groupedn2 vs direct|+0.50%|+0.76%|+6.08%|
|m512|splitk2_groupedn2 vs splitk2|+30.45%|+29.30%|+29.88%|
|m512|splitk4_groupedn2 vs splitk4|+39.67%|+43.69%|+39.84%|
|m512|pipeline vs pipeline_serial|+13.97%|+16.36%|+10.33%|
|large_n|direct_groupedn2 vs direct|-15.92%|-15.86%|-15.68%|
|large_n|splitk2_groupedn2 vs splitk2|-0.03%|-2.80%|-4.61%|
|large_n|splitk4_groupedn2 vs splitk4|-2.10%|-0.26%|-2.84%|
|large_n|splitk8_groupedn2 vs splitk8|+6.82%|+6.24%|+10.84%|
|large_n|pipeline vs pipeline_serial|+7.09%|+9.99%|+9.79%|
|m256|direct_groupedn2 vs direct|-12.81%|-12.53%|-12.81%|
|m256|splitk2_groupedn2 vs splitk2|-7.33%|-7.26%|-7.26%|
|m256|splitk4_groupedn2 vs splitk4|+21.24%|+21.22%|+21.30%|
|m256|splitk8_groupedn2 vs splitk8|+28.52%|+28.80%|+28.60%|

同进程配对仍是按候选顺序先后执行，不代表同时测量或消除了漂移。pipeline 的 pair 阶段晚于 measure，跨阶段对比仅作描述。
