# 实验代码：哪里不对、怎么改

> 对象：`pipeline/`（GitHub `ShuyaFeng/GCG-PE`）
> 配套：`../REVISION_PLAN_v3.md`（论文修改方案）、`../review_v3_findings.md`（47 条评审）
> 更新：2026-09-25

**重要前提**：产出论文现有数字的那份 harness **不在仓库里**（仓库里是旧版）。所以下面分两类：
- **B 类（已写好，直接能跑）**：我新增的模块，不依赖你们的新 harness。
- **A 类（要你改）**：现有文件里的缺陷，给了具体改法。标 ⚠️ 的是会让审稿人质疑结果有效性的。

---

## 0. 我已经写好、可以直接跑的代码

| 文件 | 作用 | 依赖 | 是否需要 GPU |
|---|---|---|---|
| `src/theory.py` | 论文 §4 每个常数的唯一来源：$H_\infty$、$m_S$、计数界、blind 界、设计规则、$k_{\rm vac}$、fluency 界 | faker, transformers(仅取 tokenizer) | 否 |
| `src/logschema.py` | v3 per-attempt 日志 schema + 校验器（会明确告诉你缺哪列、缺了会禁用哪个分析） | pandas | 否 |
| `src/stats_v3.py` | **一套**区间方法 + Prop 3 的功效/样本量 + AUC 带区间 + TPR@FPR + McNemar + ICC | numpy, scipy | 否 |
| `src/curve.py` | 容量律：无模型分位数、族比较、$\tau=(1-\alpha)m$ 拟合、水平位移、$\beta_{\rm eff}$、$\gamma_k$ | numpy, scipy | 否 |
| `src/diagnostics.py` | **copy 诊断**、ACR-on-controls、seed 方差分解、per-field、可行性表、2×2/DiD、协变量平衡、判定规则敏感性、随机记录碰撞率 | numpy, scipy, pandas | 否 |
| `src/controls.py` | 负对照生成器：**熵精确已知** + 不可能与真实标识符碰撞 + 协变量匹配 | numpy | 否 |
| `src/probes.py` | 缺失的 probe：random_restart（target-blind）、soft prompt（可加范数约束）、fluency 的 $\eta$ 记录、**不泄漏值的** fixed prompt | torch, transformers | 是 |
| `src/make_tables_v3.py` | 一条命令从日志产出全部表+图+`numbers.json` | 以上 | 否 |
| `experiments/run_audit.py` | 审计主驱动：2×2、容量扫描、全部 probe，强制同预算，可断点续跑 | nanogcg | 是 |
| `experiments/e5_nll_decomposition.py` | **记忆验证** + 比特分解 + loss-based AUC | torch | 是（很便宜） |
| `experiments/e7_blind_arm.py` | 算力匹配的 target-blind 臂（E7）/ 采样可达性（E10） | torch | 是 |
| `experiments/e9_enumerate_k1.py` | k=1 精确枚举，全文唯一的精确可达性数字 | torch | 是（约 10 分钟） |

已验证：`src/theory.py` 和 `src/curve.py` 的输出与论文里我写进去的每个数字**逐位一致**；`make_tables_v3.py` 生成的 `tab_power.tex` 与论文里手写的那张表完全相同。

### 跑的顺序

```bash
cd pipeline
pip install -r requirements.txt

# --- 第 0 步：零 GPU，先确认理论数字 ---
python -m src.theory                      # 打印论文 §4 的全部常数

# --- 第 1 步：造对照组（熵精确、无碰撞风险） ---
python -m src.controls -n 200 --seed 99991 \
    --out runs/x/data/controls.json \
    --individuals runs/x/data/individuals.json \
    --tokenizer gpt2 --ref-model distilgpt2      # 顺带打印协变量平衡表

# --- 第 2 步：记忆验证（先跑这个！它决定论文框架成不成立） ---
python experiments/e5_nll_decomposition.py \
    --model runs/x/models/gpt2-seed0 --base-model gpt2 \
    --individuals runs/x/data/individuals.json \
    --controls runs/x/data/controls.json \
    --fields ssn email --out runs/x/e5_bits.json

# --- 第 3 步：主审计 + 容量扫描（2×2 的两条臂各跑一次） ---
python experiments/run_audit.py \
    --model runs/x/models/gpt2-seed0 --model-name gpt2 --model-state finetuned \
    --individuals runs/x/data/individuals.json --controls runs/x/data/controls.json \
    --fields ssn email --probes gcg_free fixed random_restart \
    --capacities 1 2 3 4 6 8 12 16 20 24 32 48 64 \
    --restarts 3 --steps 200 --search-width 512 --decode-len 48 \
    --out runs/x/attempts_v3.jsonl

python experiments/run_audit.py \
    --model gpt2 --model-name gpt2 --model-state base \
    --individuals runs/x/data/individuals.json --controls runs/x/data/controls.json \
    --fields ssn email --probes gcg_free --capacities 6 8 20 \
    --restarts 3 --steps 200 --search-width 512 --decode-len 48 \
    --out runs/x/attempts_v3.jsonl          # 同一个日志，追加

# --- 第 4 步：零 GPU 诊断（copy 诊断是最关键的一条） ---
python -m src.logschema runs/x/attempts_v3.jsonl        # 先校验
python -m src.diagnostics runs/x/attempts_v3.jsonl --out runs/x/diag.json

# --- 第 5 步：容量律 + 出表出图 ---
python -m src.curve --sweep-csv <你的 per-k 汇总>        # 或直接看 diag.json
python -m src.make_tables_v3 runs/x/attempts_v3.jsonl --outdir ../tables/

# --- 第 6 步：便宜的 GPU 补充 ---
python experiments/e9_enumerate_k1.py --model runs/x/models/gpt2-seed0 \
    --decode-len 48 --out runs/x/e9_ft.json
python experiments/e7_blind_arm.py --model runs/x/models/gpt2-seed0 \
    --mode reachability --capacity 2 --budget 20000 --out runs/x/e10_k2.json
python experiments/e7_blind_arm.py --model runs/x/models/gpt2-seed0 \
    --individuals runs/x/data/individuals.json --controls runs/x/data/controls.json \
    --capacity 20 --budget 102400 --out runs/x/e7_blind.jsonl
```

---

## 1. ⚠️ 必须改：`src/extract_baselines.py` 的 name 字段是复读，不是提取

`_completion()` 第 40-46 行：

```python
prefixes = [
    f"{ind['name']}\n{lab}",          # field='name' 时 -> "John Smith\nName:"
    f"Name: {ind['name']}\n{lab}",     #              -> "Name: John Smith\nName:"
    ...
    f"Customer Profile\nName: {ind['name']}\nEmail: {ind.get('email','')}\n{lab}",
]
```

**问题**：探测 `name` 字段时 prompt 里已经含有 name，模型照抄即算命中，所以"baseline 对 name 100%"测的是复读能力。第 4 个变体还把 `email` 放进了 prompt，所以 email 的数也被污染。

**改法**：条件化永远不包含被打分字段自身的值。我在 `src/probes.py:fixed_prompts()` 里写好了——name 用 email 做 anchor，其它字段用 name。把 `extract_baselines.py` 的 `_completion` / `_fewshot` / `_direct` / `_template` 全部改成这个规则，或者直接换用 `probes.run_fixed`。

顺带：`_direct`/`_template` 也把 `ind['name']` 放进了 prompt，探测 name 时同样有问题。

---

## 2. ⚠️ 必须改：优化目标和打分目标是两个不同的串，而且各 probe 不一致

- `data_gen.py` 的 `FIELD_LABELS` 让 context-free GCG 的 target 是**带标签串** `"SSN: 123-45-6789"`；
- `extract_gcg.py` 的 `_anchored_target()` 返回的是**裸值**（带前导空格）；
- 两者最后都由 `utils.field_hit(out, value, field)` 对**裸值**打分。

**问题**：两个 probe 的**目标函数不同**，所以 anchored 与 context-free 的对比不是容量对比。而且论文正式定义里的 $t$ 同时被当成"优化的串"和"被打分的串"，实际不是同一个对象。

**改法**：统一成优化裸值（与打分一致）。我在 `experiments/run_audit.py:run_gcg_probe()` 里已经这么做了，并且把 `target_string` 和 `scored_value` **两列都记进日志**，这样即使将来再改也能事后核查。

---

## 3. ⚠️ 必须加：copy 诊断所需的字段（这是最大的有效性威胁）

现在 `Attempt` 记了 `best_prompt`，但分析代码从不检查 prompt 里是否含目标。

**为什么关键**：实测 forcing 在 k=1,2,3 是 0/150（寻址比特 31-47，已超 SSN 的 28.7），到 k=4-6 才打开，而 k=4-6 恰好等于目标自身的 token 长度（SSN 5-7，email 6-10）。这是 **prompt→输出复制** 的签名，不是容量的签名。如果命中主要是复制，论文的机制叙事要重写，处方也从"加第二条臂"变成"约束 probe"。

**改法**：日志里记 `prompt_text`，并在写日志时就算 `prompt_contains_target`（`run_audit.py` 已经这么做）。然后跑：

```bash
python -m src.diagnostics runs/x/attempts_v3.jsonl --only copy
```

输出里 `non_copy_rate` 就是真正配得上叫 forcing 的那个率。

---

## 4. ⚠️ 必须记：解码长度 $L$、search width、前向次数、被检视的生成次数

现在完全没记。论文两处承诺报 $L$（l.271、l.909）都没兑现，`tab:main` 的 step 数/search width 在论文和附录里一个字都没有。

**为什么关键**：
- $m_S \le c_V L - \ell_F + 1$，所以 **$L$ 不给出来，$m_S$ 就没法算，blind 界就不能引用**；
- blind 界的 $Q$ 是**被解码检视的 prompt 数**，不是前向次数。把两者混为一谈会把宣称的分离倍数夸大约 9 倍；
- random restart 说是 512 候选，但 `configs/full.yaml` 里 GCG 是 `num_steps 500 × search_width 256`，即 60-250 倍差距，论文自己承认两次却仍把 0% 那一行当头条证据。

**改法**：`AttemptV3` 里的 `decode_len_L` / `forward_passes` / `generations_inspected` 三列分开记（`logschema.py` 已定义，`run_audit.py` 已填）。`logschema.validate()` 会自动检查两条臂的 `forward_passes` 中位数是否相差超过 10%，超了就 warn。

---

## 5. ⚠️ 必须加：负对照臂本身（仓库里完全没有）

`grep -r "negative_control\|target_membership\|control_individuals" src/ run.py` → 零命中。`data_gen.py` 只用一个 `gen_seed` 生成一个群体。

**改法**：用 `src/controls.py`。它同时解决三件事：
1. **不相交 seed 流** + id 偏移 1,000,000，保证两臂永不混淆；
2. **熵精确已知**（SSN 26.56 bits、email 53.28 bits 等），所以 blind 界可以引用——Faker 的 email 我算出是 13.811 bits，而且 Monte Carlo 给不出合法的下界，只有枚举才行（`theory.py` 里做了枚举）；
3. **伦理**：SSN 用从未发放的 900-999 区段、卡号刻意 Luhn 无效、电话用保留的 555-01xx、域名用 RFC 2606 保留域。碰撞从"大概不会"变成"不可能"。

> 现在 Faker 的 `ssn()` 是从 SSA 2011 年后随机分配的约 8.89 亿号码空间里抽的，`credit_card_number()` 是 Luhn-valid 的。论文"No real personal information is used"是一个没算过的断言，而 honeytoken 提案还要让生产模型把这类串吐给用户并写进日志。这条必须改，否则伦理评审会问。

---

## 6. 必须改：`src/utils.py` 的打分规则要能被审计

`field_hit()` 现在对 numeric 字段从**输出和值两边**都剥掉 `[\s\-().]`，所以 9 位 SSN 能匹配输出里任意一段 9 位连续数字。

**问题**：这让 $m_S > 1$（一个 10 位电话号码含 2 个 9 位 SSN 窗口），而论文把 $m_S$ 留作 undefined；加上 early stopping 每步都检查，每次 attempt 有约 200 次匹配机会，把任何偶然匹配率乘了 200。

**改法**：不用改 `field_hit` 的语义（它是一个合理的规则），但要：
1. 在附录里**逐字写出**这条规则（`theory.py` 的 docstring 可以直接抄）；
2. 用 `theory.digit_multiplicity(tok, L)` 算出 $m_S$ 并报出来；
3. 用 `diagnostics.scoring_sensitivity()` 在**已有的生成文本**上重打分（不同 $L$、不剥分隔符、整输出相等），看 floor 是否稳定。我在合成日志上跑过，`whole_output_equality` 是 0.000——这就是 $\rho^=$ 退化的经验证据；
4. 用 `diagnostics.random_record_match_rate()` 把"near-zero"换成带分母和区间的数。

---

## 7. 必须改：三个 restart 不能当成 150 个独立 attempt

现在 `Attempt` 里 `seed` 是优化器 seed，但分析把 `25 人 × 2 字段 × 3 restart = 150` 当独立样本。报告的半宽 6.0 点已经接近 iid 二项的 6.7，意味着同一目标三次重启的分歧接近独立抛硬币——那 $k_{\min}(t)$ 就不是目标的属性。

**改法**：
1. 分析单元固定为 **person**，用 `stats_v3.cluster_bootstrap_rate` / `paired_bootstrap_diff`；
2. 先跑 `diagnostics.seed_variance()`。如果格子大多是 0/3 和 3/3，$k_{\min}(t)$ 有意义；如果 1/3、2/3 常见，**所有基于 $k_{\min}$ 的量（含 ACR 对打）必须放弃**；
3. 明确定义审计统计量：是 per-attempt 率还是 any-of-R 率？两者差别不小（合成数据上 0.773 vs 0.98），要在全部表里用同一个。

同时**区间方法只留一套**。现有 CSV 里混了四种，而且 k=1,2,3 的 `10.3%` 对应 $n_{\rm eff}=33.33$（设计效应 4.5），论文从未解释，也没有任何实际样本量对应它（真 Wilson 在 n=150 是 2.50%，n=25 人是 13.3%）。`stats_v3.py` 只提供一套，并且边界格子退回 person 级 Clopper-Pearson 并在 `method` 字段里说明。

---

## 8. 必须改：`src/evaluate.py` 的 `FIELD_CATEGORIES` 会产生 nan 行

```python
FIELD_CATEGORIES = {"Names":["name"], "Contact info":["email","phone"],
                    "Structured identifiers":["ssn","credit_card"], "Address":["address"]}
```
只探测了 name/email/ssn 时，`Address` 整行是 nan。旧版 `tables.py` 还有 `\end{{tabular}}` 双括号、`+{delta:.1f}` 把负数印成 `+-27.8`、nan 单元格直接输出等 bug。

**改法**：改用 `src/make_tables_v3.py`，它只输出日志里真实存在的行，并且格式化前会检查有限性。旧的 `tables.py` 建议直接停用。

---

## 9. ⚠️ 必须改：图脚本硬编码了全部数字（我已经改好）

`figures/fig_spectrum.py` 原来第 42-51 行硬编码了 8 组速率（注释还写着 "the final run, pooled"），`figures/fig_control_comparison_v2.py` 硬编码了四个模型的速率**且完全没有读 log 的代码路径**，`fig_capacity_bound_clarified` 连生成脚本都没有。而 Open Science 写的是 "No number is transcribed by hand, and re-running the script reproduces them exactly"——**这句话可以被审稿人一眼证伪**。

**已改**：两个脚本的 `--log` 现在都是必需参数，字面量全部删除，Clopper-Pearson 上界改为计算而非粘贴。`fig_capacity*` 那张图论文里已经删掉了（它画的曲线在 $k\ge2$ 恒为 1，图注自己承认"contains no empirical points"）。

---

## 10. 缺失的 probe（我已实现）

`extract_baselines.py` 只有 4 个 fixed 方法，`extract_gcg.py` 只有 GCG。论文 `tab:spectrum` 里的 8 行有 5 行在仓库里不存在：soft prompt、PII-Scope、PII-Compass、random restart、fluency-regularized。

**已实现**（`src/probes.py`）：
- `random_restart`：target-blind，支持共享 pool（blind 搜索不依赖目标，所以一个 pool 服务所有目标，这样全预算才跑得起）；
- `soft_prompt`：带 `norm_cap`。**无约束 soft prompt 两臂都 100% 是 ceiling effect，不是证据**；有范数约束才能把它接到与离散 $k$ 同一根容量轴上；
- `fluency_penalty_bits`：记录 fluency 变体的真实 $\eta$，这样 Cor fluency 才能被检验（旧版只有 Lagrangian $\lambda$，$\eta$ 从未记录）。

PII-Scope / PII-Compass 我没实现——它们针对大预训练模型上的自然 PII，在你们这个小微调语料上得 0%/0% 本身没信息量。建议移到附录一行说明，或者换成**预算匹配的、按训练版式构造的** fixed prompt 库。

---

## 11. 配置文件要改的地方

`configs/*.yaml` 全部缺 control 臂和容量扫描。新增一个：

```yaml
# configs/audit_v3.yaml
exp_name: audit_v3
output_dir: runs/audit_v3
seeds: [0, 1, 2]            # 优化器 restart
train_seeds: [0, 1, 2]      # 微调 seed —— 旧版只有 1 个，这是主要的效力短板

data:
  gen_seed: 42
  num_individuals: 200      # 旧版 30；功效计算要求 ≥400/臂才能测出 m=0.10 @ k=6
  control_seed: 99991       # 必须与 gen_seed 不相交
  num_controls: 200
  freq_map: {1: 50, 5: 75, 20: 75}

extract:
  fields: [ssn, email]
  decode_len: 48            # 必须显式写出来并报告
  capacities: [1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 32, 48, 64]
  probes: [fixed, random_restart, gcg_free, gcg_anchored, gcg_fluent, softprompt]
  gcg:
    num_steps: 200
    search_width: 512
    topk: 256
  blind:
    budget_matched: true    # Q = num_steps * search_width，不是 512
```

---

## 12. 优先级

**零 GPU，先做（一天内能出结果，而且可能改变论文结论）**
1. `python -m src.theory` → 换掉论文里错的 $H_\infty$（29.897 → 28.729）
2. copy 诊断 → 决定机制叙事（**最大威胁**）
3. ACR-on-controls → 从现有 log 直接算，可能变成主表
4. seed 方差分解 → 决定 $k_{\min}$ 类分析能不能留
5. per-field 拆分 → §4.3 定稿的前提
6. 区间统一 + 设计效应 → 所有区间在此之前都是暂定的
7. 可行性表 + 2×2 的 placebo/DiD → 两个现成的结果

**便宜 GPU，其次**
8. E5 记忆验证（**最关键**：如果真前缀补全仍是 0%，"内容被记住了"这个承重前提要重写）
9. E9 k=1 枚举、E10 k=2 采样、E7 算力匹配 blind 臂、base checkpoint 全扫描

**决定能不能中**
10. 统一审计：≥3 个**微调** seed × ≥200 人 × 一个预注册预算，每个模型带 base 臂
11. 频次 dose-response（$f=0$ 就是对照；截距=floor，斜率=memorization signal），在 $k\approx6$ 和 $k=20$ 各做一遍
12. Pythia + the Pile（**无需训练**，用 infini-gram 按 count 分桶，count=0 是天然真实对照）
13. 一个现代 instruct 模型 + LoRA

---

## 13. 一句话

仓库里现在缺的不是算力，是**对照臂、日志字段和一套统一的区间方法**。前 7 条零 GPU 的检查一天能跑完，而其中两条（copy 诊断、记忆验证）有可能直接改变论文的主叙事——所以先跑它们，再决定要不要买更多 GPU 时间。
