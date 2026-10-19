# cs336-tutor

用来学 [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/) 的 Claude Skill。

它把 Claude 变成一个苏格拉底式家教：**作业代码由你自己写**，Claude 负责在你卡住时逐级给提示、帮你看懂报错、在测试通过后 review 代码，并用回忆题、推导题和间隔重复帮你记牢课上的内容。

## 能做什么

| 你说 | Claude 做什么 |
|---|---|
| "A1 的 BPE 训练不知道从哪下手" | 按提示阶梯一级一级给提示：L0 先澄清问题 → L1 指方向 → L2 定位错误 → L3 给伪代码。不直接给答案代码 |
| "test_rope 挂了" + 报错 | 带你读报错、查 shape 和数值问题，指出哪一行有错、为什么错，修改由你自己完成 |
| "测试全过了，帮我 review" | 给出最重要的 3 个问题（正确性、数值稳定性、效率），再问一个检验理解的问题 |
| "考我 lecture 3" | 5 道题逐题问：2 道回忆、2 道推导、1 道迁移。每题先让你报信心程度，再给反馈 |
| "给我出 FLOPs 和显存的计算题" | 参数量、6ND、AdamW 每参数 16 字节、roofline、Chinchilla、GRPO advantage 等，可以换数字反复练 |
| "今天复习什么" | 读取进度日志，先安排到期的复习，再开始新内容 |
| "帮我排个学习计划" | 根据你的目标、每周可用时间和 GPU 条件排顺序，并拆成小任务 |

## 安装

### 方式一：Claude App（网页 / 桌面 / 手机）

把 `cs336-tutor` 文件夹打包成 zip（zip 里第一层就是这个文件夹，文件夹里有 `SKILL.md`），然后在 Claude 设置的 Skills 页面上传。

### 方式二：Claude Code

```bash
git clone https://github.com/Looffeeuu/cs336-tutor.git ~/.claude/skills/cs336-tutor
```

或者只给某个项目用：放到作业仓库的 `.claude/skills/cs336-tutor/` 下。

## 推荐用法

1. **在作业仓库里用 Claude Code。** Claude 能直接读你的代码和测试结果，还能把进度写进仓库根目录的 `CS336_PROGRESS.md`（模板见 `templates/`）。
2. **每次开始先说进度**，比如"继续 A1，今天做 RMSNorm 和 RoPE"。Claude 会先安排到期的复习，再推进新内容。
3. **每看完一讲，说一句"考我 lecture N"。** 不要先让它总结，先自己回忆效果更好。
4. **卡住时把报错和你的代码贴给它**，而不是直接要答案。真想要答案，它会拒绝一次，然后给你下一级提示。
5. **涉及细节时把 handout 对应段落发给它。** 课程每年都会调整，Claude 会以你给的 handout 为准。

### 推荐顺序（偏 reasoning / agentic RL 方向）

A1 完整做 → A5（SFT / expert iteration / GRPO）→ A2（profiling 和 FlashAttention 为主）→ A3、A4 以看课和做计算题为主。

## 示例对话

```
你：考我 lecture 2，关于 PyTorch 和资源计算
Claude：第 1 题（推导）：一个 1B 参数的模型用 AdamW、全 fp32 训练，
        不算激活，参数、梯度和优化器状态一共要多少显存？
        先算，再告诉我你的信心：sure / probably / guessing？
```

```
你：A1 的 BPE 太慢了，OpenWebText 跑不完
Claude（L1）：你现在每做一次 merge，是重新统计所有 pair，
        还是只更新受影响的 pair？哪些 pair 的计数会因为这次 merge 改变？
```

## 关于诚信

CS336 对使用 AI 有明确规定（以当年 handout 为准），课程方也要求不要公开作业解答。这个 skill 默认不写会被测试打分的代码。如果你把自己的作业放到 GitHub，请用**私有仓库**。

## 文件结构

```
cs336-tutor/
├── SKILL.md                    # Skill 本体（Claude 读的指令）
├── templates/
│   └── CS336_PROGRESS.md       # 进度与复习日志模板
├── README.md
└── LICENSE
```

## License

MIT
