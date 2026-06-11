# 환경 및 Source–Target Pair 설계

## 1. Simulator

### 1.1 Benchmark family

본 연구에서는 **Overcooked-AI** 계열의 cooperative cooking benchmark를 사용한다. Overcooked-AI는 두 명의 agent가 같은 team reward를 공유하며 soup를 가능한 빠르게 만들어 배달하는 fully cooperative gridworld 환경이다. 기본적인 soup completion pipeline은 다음과 같다.

1. ingredient dispenser에서 onion을 집는다.
2. onion을 pot에 넣는다.
3. pot이 완성될 때까지 기다린다.
4. dish를 이용해 완성된 soup를 집는다.
5. serving location에 soup를 배달한다.

이 환경은 coordination-induced negative transfer를 분석하기에 적합하다. 이유는 target 성능이 단일 agent의 object manipulation skill만으로 결정되지 않고, 역할 분담, handoff, 기다리기, 이동 순서, 경로 양보, 공간 분할과 같은 agent 간 coordination convention에 크게 의존하기 때문이다.

### 1.2 Main simulator

실험 구현에는 **JaxMARL Overcooked**를 사용한다. JaxMARL Overcooked는 Overcooked-AI 계열 환경을 JAX 기반으로 구현한 simulator이며, 원본 Overcooked-AI의 대표 layout들을 제공하고 custom layout 생성도 지원한다. Online MARL transfer 실험은 source와 target에서 많은 interaction을 요구하므로, vectorized execution이 가능한 JaxMARL 구현이 실험 효율 측면에서 적합하다.

다만 layout 정의와 task semantics의 기준은 Overcooked-AI를 따른다. 즉, `forced_coordination`, `coordination_ring`, `counter_circuit`, `bottleneck` 등은 Overcooked-AI layout family를 기준으로 정의하고, 실제 JaxMARL 실험에서는 동일한 station 위치, wall/counter 구조, agent start position을 보존하도록 layout을 변환한다.

모든 실험에서는 source와 target에 대해 동일한 simulator version과 동일한 environment dynamics를 사용한다. 특히 pot cooking rule은 반드시 고정해야 한다. 어떤 구현에서는 pot에 onion 3개가 들어가면 자동으로 cooking이 시작되고, 어떤 구현에서는 추가 interaction이 필요할 수 있다. 본 연구에서는 하나의 규칙을 선택한 뒤 모든 pair, 모든 method, 모든 baseline에 동일하게 적용한다.

### 1.3 기본 environment setting

| 항목 | 설정 |
|---|---|
| Number of agents | 2 |
| Agent type | Homogeneous cooperative agents |
| Reward | Shared team reward |
| Action space | 6 discrete actions: right, down, left, up, interact, no-op |
| Observation | Fully observable grid-based observation 또는 이에 대응되는 featurized observation |
| Recipe | 기본적으로 onion soup |
| Learning mode | Target task에서 online MARL |
| Evaluation | Fixed target interaction budget 하에서 비교 |

주요 성능 평가는 sparse team reward, 즉 soup delivery 성공 횟수 또는 delivery reward를 기준으로 한다. Reward shaping을 사용할 수는 있지만, 같은 pair 안에서는 모든 method와 baseline에 동일하게 적용해야 한다. 또한 shaped reward 기반 training curve와 sparse reward 기반 evaluation curve는 분리해서 보고한다.

### 1.4 Layout notation

아래 layout은 Overcooked-AI style notation을 따른다.

| Symbol | 의미 |
|---|---|
| `X` | Counter, wall, 또는 non-walkable tile |
| ` ` | Walkable floor tile |
| `O` | Onion dispenser |
| `D` | Dish dispenser |
| `P` | Pot |
| `S` | Serving location |
| `1` | Agent 1 initial position |
| `2` | Agent 2 initial position |

JaxMARL에서 실행할 때는 아래 grid를 JaxMARL custom-layout format으로 변환하되, object position, traversable cell, start position은 동일하게 유지한다.

---

## 2. Source–Target Pair 설계 원칙

Source–target pair는 다음 원칙에 따라 설계한다.

첫째, source와 target은 동일한 기본 action semantics와 cooking pipeline을 공유한다. 즉, source에서 학습된 onion pickup, pot interaction, dish pickup, soup delivery 같은 low-level skill은 target에서도 여전히 유용할 수 있어야 한다.

둘째, source와 target의 차이는 단순한 난이도 차이가 아니라 **coordination requirement의 차이**가 되도록 설계한다. Source에서 유용했던 역할 고정, handoff, waiting, ring movement convention 등이 target에서는 오히려 잘못된 inductive bias가 되도록 한다.

셋째, full source policy transfer가 scratch보다 나빠지는 경우를 단순한 state distribution shift로 해석하지 않도록 behavior-level diagnostic을 함께 측정한다. 즉, target 성능 저하가 실제로 source-style coordination pattern의 지속과 연결되는지 확인한다.

---

## 3. Pair overview

| Pair | Source layout | Target layout | Main mismatch | Tested failure mode |
|---|---|---|---|---|
| Pair A | `forced_coordination` | `forced_coordination_reversed` | 동일한 task 구조에서 agent-role assignment만 반전 | Role identity lock-in |
| Pair B | `forced_coordination` | `coordination_ring` | Fixed spatial role vs ring movement coordination | Handoff/fixed-role prior가 새로운 movement convention을 방해 |
| Pair C | `forced_coordination` | `open_parallel_room` | Handoff-based cooperation vs parallel independent cooking | Source handoff habit이 target parallelism을 억제 |
| Pair D | `bottleneck` | `wide_bottleneck` | Narrow-passage waiting vs open-passage simultaneous movement | Waiting/yielding convention이 불필요한 delay가 됨 |
| Pair E | `coordination_ring` | `counter_circuit_o_1order` | Circular movement convention vs counter/station coordination | Movement convention mismatch |

---

# Pair A. Role Identity Lock-in

## A.1 Source and target

| 구분 | Layout |
|---|---|
| Source | `forced_coordination` |
| Target | `forced_coordination_reversed` |

`forced_coordination`은 agent들이 서로 다른 공간적 역할을 수행하도록 유도하는 layout이다. Target인 `forced_coordination_reversed`는 source와 동일한 object layout, recipe, station structure를 유지하되 agent 1과 agent 2의 initial position만 바꾼다.

따라서 source와 target의 low-level cooking skill은 거의 동일하지만, agent identity에 붙은 role prior는 target에서 반대로 적용되어야 한다.

## A.2 Target layout

```python
forced_coordination_reversed = {
    "grid": """
XXXPX
O X2P
O1X X
D X X
XXXSX
""",
    "start_bonus_orders": [],
    "start_all_orders": [
        {"ingredients": ["onion", "onion", "onion"]}
    ],
    "rew_shaping_params": None,
}
```

## A.3 Coordination mismatch

Source에서는 agent index에 따라 특정 역할이 고착될 수 있다. 예를 들어 agent 1은 cooker/server, agent 2는 supplier처럼 행동하는 convention이 형성될 수 있다. Target에서는 start position이 바뀌기 때문에 동일한 역할 mapping이 더 이상 적절하지 않다.

이 pair는 다음 질문을 검증한다.

> Source에서 성공적이었던 agent-specific role prior가 target에서 반대로 요구될 때, full policy transfer는 scratch learning보다 더 나빠지는가?

## A.4 Expected negative transfer

Full transfer는 source에서 학습된 역할 구조를 그대로 유지하려고 하므로 target에서 잘못된 역할 배정을 유발할 수 있다. 이 경우 agents는 target geometry가 요구하는 새로운 role assignment를 늦게 발견하거나, source role에 고착되어 target 초기 성능이 scratch보다 낮아질 수 있다.

## A.5 Diagnostics

| Metric | 해석 |
|---|---|
| Agent-specific subtask frequency | 각 agent가 onion, dish, pot, serve 중 무엇을 담당하는지 측정 |
| Role entropy | 역할이 유연하게 바뀌는지, 특정 agent에 고착되는지 측정 |
| Time-to-first-soup | source role prior 때문에 첫 성공이 지연되는지 확인 |
| Target AUC | fixed budget 안에서 scratch보다 나쁜지 확인 |
| Role-swapped oracle | 역할 mismatch가 원인인지 확인하는 diagnostic baseline |

---

# Pair B. Fixed-role Handoff to Ring Movement Coordination

## B.1 Source and target

| 구분 | Layout |
|---|---|
| Source | `forced_coordination` |
| Target | `coordination_ring` |

`forced_coordination`은 spatially separated role specialization과 handoff-like coordination을 유도한다. 반면 `coordination_ring`은 agents가 ring-like topology 안에서 서로를 막지 않고 compatible movement convention을 형성하는 것이 중요하다.

## B.2 Coordination mismatch

Source에서 학습된 fixed-role/handoff prior는 target에서 movement-level coordination을 방해할 수 있다. Target에서는 어느 한 agent가 고정적으로 상대를 지원하거나 기다리는 것보다, 두 agent가 안정적인 circulation convention을 형성하고 blocking을 줄이는 것이 중요하다.

이 pair는 다음 질문을 검증한다.

> Source의 fixed-role 및 handoff convention이 target에서 필요한 ring movement convention 발견을 지연시키는가?

## B.3 Expected negative transfer

Full transfer agent는 source에서 배운 기다리기, 특정 agent support, handoff preference를 target에서도 유지할 수 있다. 이로 인해 ring corridor에서 blocking이 증가하거나, movement convention 형성이 늦어질 수 있다.

## B.4 Diagnostics

| Metric | 해석 |
|---|---|
| Blocking count | agents가 서로의 이동을 방해하는 빈도 |
| No-op rate | source-style waiting이 target에서도 과도하게 나타나는지 |
| Movement direction consistency | ring 안에서 안정적인 circulation convention이 형성되는지 |
| Inter-agent distance | 불필요하게 붙어 다니거나 특정 agent를 따라가는지 |
| Target AUC | early target learning efficiency 비교 |

---

# Pair C. Handoff-based Cooperation to Parallel Independent Cooking

## C.1 Source and target

| 구분 | Layout |
|---|---|
| Source | `forced_coordination` |
| Target | `open_parallel_room` |

`open_parallel_room`은 두 agent가 각자의 주변에서 onion, dish, pot에 접근할 수 있도록 구성된 layout이다. 따라서 target optimal behavior는 source-style handoff보다 parallel split execution에 가깝다.

## C.2 Target layout

```python
open_parallel_room = {
    "grid": """
XXXXXXXXX
XO P P OX
X1     2X
XD  S  DX
XXXXXXXXX
""",
    "start_bonus_orders": [],
    "start_all_orders": [
        {"ingredients": ["onion", "onion", "onion"]}
    ],
    "rew_shaping_params": None,
}
```

## C.3 Coordination mismatch

Source에서는 한 agent가 ingredient/dish supplier 역할을 하고 다른 agent가 pot/serve 역할을 하는 식의 협동이 유용할 수 있다. Target에서는 양쪽 agent가 독립적으로 cooking pipeline을 진행할 수 있으므로, source-style dependency가 오히려 parallelism을 방해한다.

이 pair는 다음 질문을 검증한다.

> Source에서 유용했던 handoff 또는 fixed-support convention이 target에서 가능한 parallel execution을 억제하는가?

## C.4 Expected negative transfer

Full transfer는 한 agent를 계속 supplier/support role에 묶고, 다른 agent를 finisher role에 묶을 수 있다. 이 경우 target에서 두 agent가 동시에 soup-making progress를 내지 못하고, 불필요한 cross-room movement나 waiting이 증가한다.

## C.5 Diagnostics

| Metric | 해석 |
|---|---|
| Simultaneous progress rate | 두 agent가 동시에 target-relevant subtask를 수행하는 정도 |
| Handoff count | local resource가 있는데도 handoff를 시도하는지 |
| Role entropy | 역할이 과도하게 고정되는지 |
| Distance traveled per soup | 불필요한 cross-room movement가 증가하는지 |
| Time-to-first-soup | source habit 때문에 첫 성공이 늦어지는지 |
| Target AUC | full transfer가 scratch보다 나쁜지 |

---

# Pair D. Bottleneck Waiting to Open-passage Movement

## D.1 Source and target

| 구분 | Layout |
|---|---|
| Source | `bottleneck` |
| Target | `wide_bottleneck` |

`bottleneck`은 좁은 통로 때문에 agents가 서로 양보하거나 기다리는 convention을 학습하기 쉬운 layout이다. `wide_bottleneck`은 source의 station structure를 유지하되 중앙 통로를 넓혀 simultaneous movement가 가능하도록 만든다.

## D.2 Target layout

```python
wide_bottleneck = {
    "grid": """
XXOXDXX
X 1 2 X
X     X
X     X
XSXXPPX
""",
    "start_bonus_orders": [],
    "start_all_orders": [
        {"ingredients": ["onion", "onion", "onion"]}
    ],
    "rew_shaping_params": None,
}
```

## D.3 Coordination mismatch

Source에서는 bottleneck 근처에서 기다리거나 양보하는 행동이 blocking을 줄이는 좋은 coordination pattern일 수 있다. Target에서는 통로가 넓어졌기 때문에 과도한 waiting은 더 이상 필요하지 않으며, 오히려 soup completion을 늦춘다.

이 pair는 다음 질문을 검증한다.

> Source에서 유용했던 waiting/yielding convention이 target에서는 불필요한 delay로 작용하는가?

## D.4 Expected negative transfer

Full transfer agent는 target에서도 former bottleneck region 근처에서 멈추거나 상대가 지나가기를 기다릴 수 있다. 이 경우 blocking 자체는 줄어들 수 있지만, simultaneous movement가 줄어들어 target reward가 낮아진다.

## D.5 Diagnostics

| Metric | 해석 |
|---|---|
| No-op rate near former bottleneck | source-style waiting이 유지되는지 |
| Waiting duration | 불필요하게 오래 기다리는지 |
| Simultaneous movement ratio | 두 agent가 동시에 움직이는 비율 |
| Blocking count | 실제로 기다릴 필요가 있었는지 확인 |
| Soup completion time | waiting convention 때문에 delivery가 지연되는지 |

---

# Pair E. Ring Movement Convention to Counter/Station Coordination

## E.1 Source and target

| 구분 | Layout |
|---|---|
| Source | `coordination_ring` |
| Target | `counter_circuit_o_1order` |

`coordination_ring`은 ring-like movement convention을 요구한다. Agents는 같은 방향으로 순환하거나 서로를 막지 않는 이동 규칙을 발견해야 한다. 반면 `counter_circuit_o_1order`는 onion-only target variant로, counter usage, station access, object sequencing이 상대적으로 중요하다.

## E.2 Coordination mismatch

Source에서 학습된 circular movement convention은 target에서 필요한 counter/station coordination과 맞지 않을 수 있다. Target에서는 단순히 안정적인 순환 경로를 유지하는 것보다 onion, dish, pot, serving location에 대한 효율적인 접근과 counter usage가 중요하다.

이 pair는 다음 질문을 검증한다.

> Source의 ring movement convention이 target의 station/counter-based coordination을 방해하는가?

## E.3 Expected negative transfer

Full transfer agent는 source에서 학습한 movement route를 target에서도 과도하게 보존할 수 있다. 이 경우 station visit sequence가 비효율적이 되고, counter 주변 idle time이 증가하며, soup completion이 늦어질 수 있다.

## E.4 Diagnostics

| Metric | 해석 |
|---|---|
| Counter usage efficiency | counter interaction이 target progress에 기여하는지 |
| Station visit sequence | onion, dish, pot, serve 접근 순서가 효율적인지 |
| Route efficiency | station까지의 이동 경로가 불필요하게 긴지 |
| Idle time around counters | counter 근처에서 정체되는지 |
| Target AUC | full transfer가 scratch보다 early learning에서 불리한지 |

---

## 4. Training protocol

각 pair에 대해 다음 절차를 사용한다.

### 4.1 Source training

먼저 source layout에서 source policy를 online MARL로 학습한다. Backbone은 모든 pair와 baseline에서 동일하게 유지한다. 기본 후보는 MAPPO 또는 IPPO이며, centralized training with decentralized execution 구조를 사용할 수 있다.

Source checkpoint에는 다음 정보를 함께 저장한다.

- source layout name,
- simulator version,
- environment dynamics setting,
- source training seed,
- source training budget,
- actor/critic parameters,
- encoder parameters,
- reward shaping configuration,
- episode horizon.

### 4.2 Target online transfer

Target layout에서는 fixed interaction budget \(B\) 안에서 online learning curve를 비교한다. 모든 method는 동일한 target budget, 동일한 seed set, 동일한 reward setting, 동일한 network capacity를 사용한다.

비교할 baseline은 다음과 같다.

| Method | 목적 |
|---|---|
| Scratch | Source knowledge 없이 target에서 처음부터 학습 |
| Full policy transfer | Source policy 전체 전이가 negative transfer를 유발하는지 확인 |
| Encoder-only transfer | Representation reuse와 coordination reuse를 분리 |
| Skill-only transfer | Individual skill은 유용한지 확인 |
| Coordination-prior transfer | Source coordination pattern만의 효과를 확인 |
| Proposed selective transfer | Harmful coordination prior를 online으로 억제할 수 있는지 확인 |
| Role-swapped oracle | Pair A에서 role identity mismatch를 확인하는 diagnostic |

---

## 5. Evaluation metrics

### 5.1 Main negative-transfer score

Target interaction budget \(B\) 안에서 scratch 대비 transfer의 early learning efficiency를 비교한다.

\[
\mathrm{CNT}(B)
=
\mathrm{AUC}_{\text{scratch}}(B)
-
\mathrm{AUC}_{\text{transfer}}(B).
\]

\[
\mathrm{CNT}(B)>0
\]

이면 해당 transfer method가 target online learning 구간에서 scratch보다 나쁜 성능을 보인 것이다.

### 5.2 Pair-specific behavior diagnostics

Reward만으로는 coordination-induced negative transfer를 충분히 설명할 수 없다. 따라서 각 pair의 failure mode에 맞는 behavior diagnostic을 함께 보고한다.

| Pair | 필수 diagnostic |
|---|---|
| Pair A | agent-specific role frequency, role entropy, role-swapped oracle comparison |
| Pair B | blocking count, no-op rate, movement direction consistency, circulation stability |
| Pair C | simultaneous progress rate, handoff count, role entropy, distance traveled per soup |
| Pair D | no-op rate near former bottleneck, waiting duration, simultaneous movement ratio |
| Pair E | counter usage efficiency, station visit sequence, route efficiency, idle time around counters |

---

## 6. Expected evidence pattern

이 source–target suite가 의도대로 작동한다면 다음과 같은 결과 패턴을 기대한다.

1. Full source-policy transfer는 coordination requirement가 유사한 경우에는 유용하거나 최소한 큰 손해를 주지 않는다.
2. Coordination requirement가 다른 pair에서는 full source-policy transfer가 scratch보다 낮은 early target AUC를 보일 수 있다.
3. Encoder-only 또는 skill-only transfer가 full transfer보다 좋은 경우, negative transfer의 원인이 source knowledge 전체가 아니라 source-specific coordination prior임을 보여준다.
4. Proposed selective transfer는 harmful source coordination pattern을 약화하고, target에서 필요한 새로운 coordination을 더 빠르게 발견해야 한다.
5. Behavior diagnostic에서는 full transfer agent가 source-style behavior를 오래 유지하고, proposed method는 target learning 과정에서 해당 behavior를 줄이는 패턴이 나타나야 한다.

---

## 7. Reporting checklist

논문 또는 실험 보고서에는 다음 정보를 반드시 명시한다.

- simulator name and version,
- source layout and target layout,
- built-in layout인지 custom layout인지,
- exact custom layout grid,
- pot cooking rule,
- episode horizon,
- source training budget,
- target interaction budget,
- number of random seeds,
- sparse reward definition,
- reward shaping parameters,
- policy parameter sharing 여부,
- agent identity observation 사용 여부,
- centralized training / decentralized execution 여부,
- target evaluation protocol,
- behavior diagnostic 계산 방식.

---

## 8. References

- Overcooked-AI repository: <https://github.com/HumanCompatibleAI/overcooked_ai>
- JaxMARL Overcooked documentation: <https://jaxmarl.foersterlab.com/environments/overcooked/>
- `forced_coordination.layout`: <https://raw.githubusercontent.com/HumanCompatibleAI/overcooked_ai/master/src/overcooked_ai_py/data/layouts/forced_coordination.layout>
- `coordination_ring.layout`: <https://raw.githubusercontent.com/HumanCompatibleAI/overcooked_ai/master/src/overcooked_ai_py/data/layouts/coordination_ring.layout>
- `counter_circuit_o_1order.layout`: <https://raw.githubusercontent.com/HumanCompatibleAI/overcooked_ai/master/src/overcooked_ai_py/data/layouts/counter_circuit_o_1order.layout>
- `bottleneck.layout`: <https://raw.githubusercontent.com/HumanCompatibleAI/overcooked_ai/master/src/overcooked_ai_py/data/layouts/bottleneck.layout>
