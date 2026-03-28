可以，而且我觉得**很适合从“最近参考帧”下手**。

先说判断：DreamZero 是自回归式 WAM，推理时会把过去的干净观测放进 KV cache，并在每次动作块执行后把预测帧替换成真实观测重新写回 cache；论文也明确说它的主要延迟瓶颈仍然是 **DiT 大骨干 + 多步去噪**，单纯“只生成动作不生成视频”提速并不明显。DreamZero 当前已经用了几类推理优化：CFG 并行、DiT caching（把有效步数从 16 降到 4）、torch.compile/CUDA Graphs、PTQ；而 DreamZero-Flash 把 4 步进一步压到 1 步是**训练后的**方案，不属于你现在说的 train-free 范畴。基于这些信息，我的判断是：**参考帧 token 删减很值得做，但它更像“缩短上下文注意力长度、减小 KV cache、降低每步 DiT 代价”的补充加速项，最好和现有 DiT caching 一起上，而不是指望它单独解决 DreamZero 的大头延迟。** ([arXiv][1])

## 我最推荐你的思路：把 DUET 改成 DreamZero 版

DUET 的核心是两段式：先做**视觉端去冗余压缩**，再做**语义相关的逐层剪枝**。放到 DreamZero 里，我建议你不要照搬“salient text token”，而是把第二阶段的 query 改成 **当前 noisy chunk token + action token + proprio/state token + 末端执行器空间先验**。因为 DreamZero 的当前决策真正依赖的是“当前去噪状态下，为了预测下一段动作，历史视觉里哪些区域有用”，这和 VLM 里“文本问题在问什么”是对应关系。DUET 本身就是先做 vision-only compression，再做 layer-wise query-aware dropping；DreamZero 则天然有“cache 里的历史视觉”和“当前 chunk/action/state 查询”这两种信号。 ([arXiv][2])

我会把它写成一个 **DreamZero-DUET-lite**：

[
\text{score}_i
==============

\alpha, s_i^{self}
+
\beta, s_i^{q\rightarrow ref}
+
\gamma, \Delta_i
+
\delta, r_i^{ee}
]

其中：

* (s_i^{self})：最近参考帧内部的 self-attention 入注意力分数，负责找“本来就重要”的视觉 token
* (s_i^{q\rightarrow ref})：当前 noisy 视频 token、action token、state token 对参考帧 token 的注意力，负责找“当前动作决策真要用到”的 token
* (\Delta_i)：最近两次真实观测之间该 patch 的变化强度
* (r_i^{ee})：它离末端执行器 / 抓取区域 / 接触区域有多近

然后做两步：

1. **写入 cache 前先压一遍**：保留 top-k dominant tokens，剩下 token 只做局部 merge，不做全局平均。
2. **进 DiT 后再分层删一遍**：只在若干中后层做 progressive prune，比如 1.0 → 0.7 → 0.4，而不是一开始就砍死。

这个设计同时借了 DUET、VisionZip、PyramidDrop 的长处，但把“text relevance”换成了“当前动作相关性”。DUET 和 VisionZip 说明了“先做结构冗余压缩再进大模型”是合理的，PyramidDrop 则说明了“越深层视觉冗余越大，分层删通常比一开始猛删更稳”。 ([arXiv][2])

---

## 我建议你优先尝试的 6 个 train-free 方向

### 1）只对“最近参考帧”做两阶段压缩，不碰更老的记忆帧

这是我最建议先试的起点。

原因很简单：DreamZero 的 cache 是滚动更新的，最新真实观测最频繁参与后续推理，也是你最有把握做“语义相关剪枝”的地方。老帧先别动，避免把真正的长时记忆删坏。DreamZero 本身就强调真实观测会持续写回 KV cache，所以最近参考帧是最自然、也最安全的切入点。 ([arXiv][1])

一个稳妥起手配置可以是：

* 最近参考帧原始 token 数为 (N)
* 写 cache 前保留 (0.25N) 个 dominant token
* 再用局部聚合构 0.10N～0.15N 个 contextual token
* 于是 cache 里最近参考帧先压到 35%～40%
* 进 DiT 后只在第 1/3 和 2/3 深度各 prune 一次，比如保留比例 1.0 → 0.75 → 0.5

这版很像 DUET，但更偏保守，适合你先看 success rate / action smoothness 会不会掉。

---

### 2）做“变化区域优先”的 temporal delta pruning

这条路和 DreamZero 很匹配，因为 DreamZero 在 closed-loop 下每次都拿到**真实新观测**并写回 cache。既然连续两帧很多区域通常没变，那你完全可以做 train-free 的 **change-only cache refresh**：不变区域复用旧 token 或只保留极少 token，变化区域才给足 token 预算。这个思路和 DreamZero 已经在做的 DiT caching 本质很像：论文里就是利用 successive velocity 的方向一致性，在相似时复用缓存结果。你可以把这个“复用精神”从去噪步推广到视觉 cache。 ([arXiv][1])

一个非常实用的版本是：

* 对最近观测和上一次 cache 中的真实观测做 patch-level cosine distance
* 小于阈值的 patch 不更新或只保留 1 个 merged token
* 大于阈值的 patch 保留 full token 或高预算 token
* 对机器人手爪、目标物周边设保底不删区

这种方法在“走向目标、运输中、背景基本静止”的阶段会很赚。

---

### 3）不要 one-shot 永久删除，要做 progressive prune，最好带一次“复活机会”

PyramidDrop 的结论是：浅层需要更多视觉 token，深层冗余才更明显；SwiftVLM 则进一步指出，**浅层看起来不重要的 token，深层可能变重要**，所以过早、不可逆地删掉容易伤细节任务。这个教训放在 DreamZero 上我觉得尤其重要，因为抓取接触、遮挡边界、小物体角点这些信息，往往不是一开始就“显眼”，而是随着当前 chunk 去噪深入才变重要。 ([arXiv][3])

所以我会建议你：

* 前几层不删，只压写 cache 前的 token 数
* 中层开始删
* 深层再删一轮
* 对被删 token 不是真的彻底扔掉，而是放一个 side pool
* 在某一层允许“复活一次”重新评分

你可以把它理解成 **DreamZero 版的 bypass**。这对避免“小目标本来没啥 attention，后面突然需要精细操作时却已经被删没了”非常有用。

---

### 4）别只看一种分数，做 saliency fusion

ConsensusDrop 的核心结论是：只看视觉侧 saliency 不够，只看跨模态 saliency 也不够，把两者融合通常更稳。DreamZero 里虽然不是典型 VLM 问答，但你同样有两类信号：

* **视觉结构重要性**：参考帧内部 self-attention、局部显著区域
* **当前决策相关性**：当前 noisy chunk / action / state 对参考帧 token 的关注

所以比起“只按 self-attention 排序”或者“只按当前 query-attention 排序”，我更建议你直接做 fusion score。ConsensusDrop 的实验结论正支持这种做法。 ([arXiv][4])

我的建议是别上来就手调很复杂的 learned fusion，先用简单加权：

[
\text{score}_i = 0.4,s_i^{self}+0.4,s_i^{q\to ref}+0.2,\Delta_i
]

如果你的机械臂末端位姿拿得到，再额外加一个末端 ROI bonus 就够了。

---

### 5）不要固定预算，做 complexity-aware 的自适应 token budget

固定保留 128 个、160 个这类预算，在 DreamZero 场景里大概率不是最优。静态背景阶段和精细抓取阶段的需求完全不同。E-AdaPrune 的核心观点就是：不同输入的信息密度差异很大，token budget 应该自适应，而不是所有样本同一个 top-k。DyMU 也强调根据图像复杂度动态调节压缩强度。 ([arXiv][5])

放到 DreamZero，我建议最简单的自适应信号就两个：

* **运动能量**：最近两帧差异越大，预算越高
* **接触风险**：机械臂末端离目标越近，预算越高

一个非常工程化的规则：

* 远距离接近阶段：最近参考帧保留 25%～35%
* 目标周边操作阶段：保留 50%～70%
* 抓取闭合 / 插接 / 放置末段：临时回到 80%～100%

这类 heuristic 虽然朴素，但通常比全程固定预算更稳。

---

### 6）做 action-sensitive pruning，而不是纯视觉显著性 pruning

PIO-FVLM 的思路我觉得对你很有启发：它不是只看 token 相似性，而是尽量根据“对最终输出有没有影响”来决定保留谁。它在 VLM 上是 training-free，并且兼容高效注意力实现。 ([arXiv][6])

你可以把这个思想改成 DreamZero 版：

* 每隔几次控制循环，而不是每一步
* 临时 mask 掉最近参考帧的一小组 token
* 看预测 action velocity / action chunk 的变化量
* 对动作影响越大的 token 组，优先保留
* 对动作几乎没影响的 token 组，优先删掉或 merge 掉

这会比“只保留看起来显眼的区域”更贴近你的最终目标：**动作不能变差**。
代价是它比纯 attention 排序更贵，所以适合低频做，比如每 5～10 个 chunk 重打一次分数，而不是每步都做。

---

## 除了 token 删减，我建议你同步看的 train-free 加速点

这部分不是你问题的核心，但在 DreamZero 里很重要，因为论文已经说明大头延迟来自 DiT 和去噪步数。你若只做 token pruning，不一定吃满收益。DreamZero 现有的推理侧优化包括 CFG 并行、DiT caching、torch.compile/CUDA Graphs、PTQ、kernel/scheduler 优化；其中 Flash 是训练式，不算 train-free。 ([arXiv][1])

所以我会建议你把实验分成两条线同时做：

* **线 A：token 线**
  最近参考帧两阶段压缩 + progressive prune + 自适应预算
* **线 B：system 线**
  更细调 DiT caching 阈值、确保 static shape 友好、把 token 数变化设计成少量离散档位，避免频繁 shape 抖动影响 compile/cuda graph 效果

后面这句是我的工程判断：**如果你的 token 数每步都乱变，编译图和 cache 命中会变差，部分加速可能被抵消。**所以实际部署时最好把 token budget 离散成几档，例如 64 / 96 / 128 / full。

---

## 我最建议你先做的落地顺序

第一版，不要贪多：

1. **只剪“最近参考帧”**，其余历史帧先不动。
2. **写 cache 前做 DUET 的第一阶段**：dominant keep + local merge。
3. **进 DiT 后只做一轮中层 prune**，不要多层先上。
4. **评分只用三项**：self-attn + current-query attn + frame delta。
5. **设三条硬保护规则**：

   * 末端执行器附近 token 不删
   * 高运动 token 不删
   * 至少保留若干全局 anchor token

我觉得你最有希望 first win 的，不是极限压缩，而是这种保守配置：

* 最近参考帧 token 先压到原来的 **40%～60%**
* 中层再压到 **30%～40%**
* 只在静态/低运动 chunk 开 aggressive pruning
* 一进入接触操作就自动回升预算

这版最有机会在几乎不改策略行为的前提下，先拿到一波真实延迟下降。

---

## 两个你一定要防的坑

**第一，不能只按“视觉显著”删。**
抓取任务里很多关键 token 不显眼，但对接触几何、边缘、遮挡关系很关键。VScan、SwiftVLM 这类工作都在提醒一个事实：单一 saliency 信号很容易错删任务相关的小细节。 ([arXiv][7])

**第二，别一开始就删当前观测 token。**
你现在最稳的切入点是“最近参考帧”，不是“当前输入帧”。因为 DreamZero 当前 chunk 本来就在 noisy 去噪，当前观测再 aggressively 压，很容易把动作条件一起压坏。这是我的方法判断，但和 DreamZero 里“动作预测对当前视频条件仍然敏感、少步去噪时视频条件本身就会偏 noisy”的结论是一致的。 ([arXiv][1])

---

如果你愿意，我下一条可以直接给你写一版**“DreamZero 最近参考帧 train-free token pruning 的具体算法草案”**，包括：

* 在 cache update 阶段怎么算分
* 在 DiT 哪几层 prune
* top-k 和 merge 比例怎么设
* 以及你应该怎么做 ablation。

[1]: https://arxiv.org/html/2602.15922v1 "World Action Models are Zero-shot Policies"
[2]: https://arxiv.org/abs/2602.18846 "[2602.18846] DUET-VLM: Dual stage Unified Efficient Token reduction for VLM Training and Inference"
[3]: https://arxiv.org/html/2410.17247v1 "PyramidDrop: Accelerating Your Large Vision-Language Models via Pyramid Visual Redundancy Reduction"
[4]: https://arxiv.org/html/2602.00946v1 "ConsensusDrop: Fusing Visual and Cross-Modal Saliency for Efficient Vision Language Models"
[5]: https://arxiv.org/html/2603.05950v1?utm_source=chatgpt.com "Energy-Driven Adaptive Visual Token Pruning for Efficient ..."
[6]: https://arxiv.org/abs/2602.04657 "[2602.04657] PIO-FVLM: Rethinking Training-Free Visual Token Reduction for VLM Acceleration from an Inference-Objective Perspective"
[7]: https://arxiv.org/html/2505.22654v2 "VScan: Rethinking Visual Token Reduction for Efficient Large Vision-Language Models"
