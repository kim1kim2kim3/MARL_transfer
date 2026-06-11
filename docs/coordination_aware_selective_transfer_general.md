# Coordination-Aware Selective Transfer for Cooperative MARL

## 1. Problem Statement

Online multi-agent reinforcement learning에서 새로운 cooperative task를 처음부터 학습하는 것은 많은 interaction cost를 요구한다. 따라서 이전 source task에서 학습된 지식, 행동 전략, representation, skill을 target task로 전이하여 sample efficiency를 높이려는 접근이 자연스럽게 등장한다.

그러나 cooperative MARL에서 전이되는 지식은 단일 agent의 action preference에만 국한되지 않는다. 여러 agent가 source task에서 반복적으로 상호작용하면서 형성한 coordination pattern 역시 함께 전이된다. 이러한 pattern은 함께 이동하기, 특정 agent를 따라가기, 병목 지점에서 기다리기, 특정 역할을 고정하기, 특정 agent가 특정 subtask를 담당하기, joint trajectory를 일정한 형태로 유지하기 등의 형태로 나타난다.

Source task에서 효과적이었던 coordination pattern은 target task가 유사한 협동 구조를 가질 때 학습을 가속할 수 있다. 반면 target task가 다른 역할 분담, 다른 spatial dependency, 다른 timing constraint, 다른 agent interaction structure를 요구할 경우, source coordination pattern은 유용한 prior가 아니라 harmful inductive bias가 될 수 있다. 이 경우 agents는 target task가 요구하는 새로운 coordination mode를 발견하지 못하거나, 잘못된 역할 구조에 고착되거나, joint exploration이 제한되어 scratch learning보다 낮은 초기 성능을 보일 수 있다.

본 방법론은 이러한 현상을 **coordination-induced negative transfer**로 정의하고, source knowledge를 선택적으로 재사용하는 coordination-aware transfer framework를 제안한다.

---

## 2. Core Research Question

본 연구의 핵심 질문은 다음과 같다.

> How can cooperative MARL agents reuse transferable knowledge from source tasks while avoiding harmful transfer of source-specific coordination patterns?

이를 다음의 하위 질문으로 분해한다.

1. Source task에서 학습된 지식을 individual skill과 coordination prior로 어떻게 분리할 것인가?
2. Target task의 online interaction을 통해 어떤 source coordination pattern이 유용하거나 해로운지 어떻게 추정할 것인가?
3. Target task에서 새로운 coordination을 발견할 자유도를 유지하면서 source knowledge를 어떻게 선택적으로 재사용할 것인가?
4. Negative transfer가 individual policy mismatch 때문인지, coordination pattern mismatch 때문인지 어떻게 분해하여 측정할 것인가?

---

## 3. Problem Formulation

Cooperative MARL task를 fully cooperative Markov game 또는 Dec-POMDP로 정의한다.

\[
\mathcal{M} = \langle \mathcal{S}, \{\mathcal{O}_i\}_{i=1}^{N}, \{\mathcal{A}_i\}_{i=1}^{N}, P, R, \gamma \rangle
\]

여기서 \(N\)은 agent 수, \(s_t \in \mathcal{S}\)는 global state, \(o_{i,t} \in \mathcal{O}_i\)는 agent \(i\)의 observation, \(a_{i,t} \in \mathcal{A}_i\)는 agent \(i\)의 action이다. Joint action은 다음과 같다.

\[
\mathbf{a}_t = (a_{1,t}, \dots, a_{N,t})
\]

모든 agent는 shared team reward를 받는다.

\[
r_t = R(s_t, \mathbf{a}_t)
\]

Source task와 target task는 각각 다음과 같이 표시한다.

\[
\mathcal{M}_S, \quad \mathcal{M}_T
\]

Source에서 학습된 team policy는 다음과 같다.

\[
\pi^S = \{\pi^S_1, \dots, \pi^S_N\}
\]

Target에서 학습할 policy는 다음과 같다.

\[
\pi^T = \{\pi^T_1, \dots, \pi^T_N\}
\]

목표는 target interaction budget \(B\) 안에서 다음의 expected return 또는 learning curve AUC를 최대화하는 것이다.

\[
J_T(\pi^T) = \mathbb{E}_{\tau \sim \pi^T} \left[ \sum_{t=0}^{T} \gamma^t r^T_t \right]
\]

---

## 4. Coordination-Induced Negative Transfer

Naive transfer는 source policy 전체를 target policy의 initialization, behavior prior, distillation target, regularization target으로 사용한다.

\[
\pi^T \leftarrow \pi^S
\]

또는 target 학습 중 source policy와의 거리를 줄이는 regularization을 추가한다.

\[
\mathcal{L}_{\text{transfer}} = D_{\mathrm{KL}}\left( \pi^T(\cdot \mid o) \Vert \pi^S(\cdot \mid o) \right)
\]

이 방식은 source knowledge가 target에서도 유용하다는 가정에 기반한다. 그러나 cooperative MARL에서는 source policy가 단순한 individual action rule이 아니라 source-specific coordination convention을 포함한다. 따라서 source와 target의 coordination requirement가 다르면 다음 현상이 발생할 수 있다.

- joint exploration이 source-style trajectory distribution에 갇힘
- agent-specific role이 target에서도 고정됨
- target에서 필요한 role switching 또는 dynamic allocation이 지연됨
- source에서 유효했던 waiting, following, grouping, handoff, yielding pattern이 target에서 불필요한 delay를 유발함
- target 초기 성능이 scratch learning보다 낮아짐

이를 다음과 같이 정의한다.

\[
\mathrm{CNT}(B) = \mathrm{AUC}_{\text{scratch}}(B) - \mathrm{AUC}_{\text{transfer}}(B)
\]

\[
\mathrm{CNT}(B) > 0
\]

이면 interaction budget \(B\)에서 coordination-induced negative transfer가 발생한 것으로 본다.

---

## 5. Proposed Framework: Coordination-Aware Selective Transfer

제안 방법의 핵심은 source policy를 하나의 monolithic policy로 전이하지 않고, 다음 구성요소로 분해하여 target에서 선택적으로 사용하는 것이다.

\[
\text{Source Knowledge}
=
\text{Representation}
+
\text{Individual Skill}
+
\text{Role Prior}
+
\text{Coordination Pattern}
\]

Target policy는 source knowledge를 soft prior로 사용하되, target에서 관찰되는 online evidence에 따라 source coordination prior의 영향력을 조절한다.

전체 framework는 다음 네 가지 원칙을 따른다.

### 5.1 Reuse Transferable Individual Skills

Source에서 학습된 low-level 또는 mid-level individual skill은 target에서도 유용할 수 있다. 예를 들어 navigation, object interaction, local control, subgoal reaching, obstacle avoidance, local path planning, primitive manipulation skill은 target task가 달라져도 재사용 가능성이 높다.

이러한 skill은 target policy의 encoder 또는 skill branch로 전이한다.

### 5.2 Treat Coordination Patterns as Conditional Priors

Source coordination pattern은 항상 적용되는 hard constraint가 아니라, target에서 유용성이 검증되어야 하는 conditional prior로 취급한다.

\[
\text{coordination prior} \neq \text{fixed target behavior}
\]

Target 학습 중 각 coordination pattern이 target return에 미치는 영향을 추정하고, 유용한 pattern은 유지하며, 해로운 pattern은 약화한다.

### 5.3 Preserve a Source-Free Exploration Branch

Target task가 source와 다른 coordination structure를 요구할 수 있으므로, source pattern에 의존하지 않는 free target branch를 항상 유지한다.

\[
\pi^T = \pi^{\text{free}} + \text{source-guided prior}
\]

이 branch는 source coordination library에 없는 새로운 target-specific coordination을 발견하기 위한 escape route 역할을 한다.

### 5.4 Adapt Role-Agent Mapping

Source에서 특정 agent identity에 특정 역할이 붙었더라도, target에서는 동일한 identity-role mapping이 유효하다고 가정하지 않는다. Agent-role correspondence는 target context와 online return signal을 기반으로 적응적으로 선택한다.

---

## 6. Model Architecture

제안 모델은 다음 구성요소로 이루어진다.

```text
Observation o_i
     │
     ▼
Shared / Agent Encoder E_θ
     │
     ▼
Agent representation h_i
     │
     ├── Free Target Policy Branch π_i^free
     │
     ├── Individual Skill Branch π_i^skill
     │
     ├── Source Coordination Prior Heads {π^S_{i,z}}
     │
     ├── Role / Permutation Adapter P
     │
     └── Negative-Aware Coordination Gate g_ψ
             │
             ▼
Final Target Policy π_i^T
```

---

## 7. Encoder

각 agent의 observation을 latent representation으로 변환한다.

\[
h_{i,t} = E_\theta(o_{i,t})
\]

동일한 action semantics와 observation semantics를 공유하는 task family에서는 source encoder를 target 초기화에 사용할 수 있다.

\[
E_\theta^T \leftarrow E_\theta^S
\]

Encoder는 target 학습 중 fine-tuning된다.

---

## 8. Individual Skill Module

Individual skill module은 agent별 reusable behavior primitive를 나타낸다.

\[
u_{i,t} \sim q_\xi(u \mid \tau^i_{t-K:t})
\]

여기서 \(u\)는 individual skill latent variable이고, \(\tau^i_{t-K:t}\)는 agent \(i\)의 local trajectory window이다.

Skill-conditioned policy는 다음과 같다.

\[
\pi_i^{\text{skill}}(a_i \mid h_i, u)
\]

Skill module은 target에서 다음과 같이 사용된다.

\[
\ell_i^{\text{skill}} = f_{\text{skill}}(h_i, u)
\]

이 module은 source에서 학습한 reusable control knowledge를 제공하지만, agent 간 coordination을 강제하지 않는다.

---

## 9. Source Coordination Library

Source trajectory에서 joint coordination mode를 latent variable로 추출한다.

\[
z_t \sim q_\phi(z \mid \tau^{\text{joint}}_{t-K:t})
\]

여기서

\[
\tau^{\text{joint}}_{t-K:t}
=
(s_{t-K:t}, \mathbf{o}_{t-K:t}, \mathbf{a}_{t-K:t}, r_{t-K:t})
\]

이고 \(z_t\)는 source task에서 관찰된 joint behavior mode를 나타낸다.

Coordination mode \(z\)는 다음과 같은 추상적 협동 구조를 표현할 수 있다.

- agents move together
- agents split spatially
- one agent follows another
- one agent waits while another moves
- agents use fixed roles
- agents switch roles dynamically
- agents perform sequential handoff-like dependency
- agents execute parallel subtasks
- agents maintain formation
- agents coordinate around a bottleneck or shared resource

각 coordination mode에 대해 source coordination prior head를 학습한다.

\[
\pi^S_{i,z}(a_i \mid h_i)
\]

이 head는 mode \(z\)가 활성화되었을 때 source에서 agent \(i\)가 취하던 action prior를 제공한다.

---

## 10. Role / Permutation Adapter

Agent identity와 functional role을 분리하기 위해 role adapter를 사용한다. \(N\)명의 agent가 있을 때 role mapping은 permutation 또는 soft assignment matrix로 표현한다.

Hard permutation version:

\[
P_t \in \Pi_N
\]

여기서 \(\Pi_N\)은 \(N\)개 agent에 대한 permutation set이다.

Soft assignment version:

\[
P_t \in [0,1]^{N \times N}
\]

\[
\sum_j P_{ij}=1, \quad \sum_i P_{ij}=1
\]

Role adapter는 target context와 현재 joint state를 기반으로 source role을 target agent에 매핑한다.

\[
P_t \sim p_\psi(P \mid h_{1,t}, \dots, h_{N,t}, c_T)
\]

여기서 \(c_T\)는 target task context embedding이다. 이 embedding은 target의 초기 trajectory, observation statistics, transition dynamics, reward feedback 등으로부터 추정한다.

\[
c_T = f_\eta(D_T^{\text{early}})
\]

---

## 11. Negative-Aware Coordination Gate

Coordination gate는 현재 target state에서 어떤 source coordination prior를 어느 정도 사용할지 결정한다.

\[
g_\psi(z \mid h_{1,t}, \dots, h_{N,t}, c_T)
\]

각 coordination mode \(z\)에 대해 target에서의 usefulness score를 유지한다.

\[
M_z \leftarrow (1-\rho)M_z + \rho \cdot \widehat{A}^{(z)}_t
\]

여기서 \(\widehat{A}^{(z)}_t\)는 mode \(z\)가 사용되었을 때의 target advantage 추정치이다.

\[
\widehat{A}_t = \widehat{R}_{t:t+H} - V_\omega(s_t)
\]

Mode-specific transfer strength는 다음과 같이 계산한다.

\[
\alpha_z = \alpha_{\max} \cdot \sigma\left(\frac{M_z}{\tau}\right)
\]

따라서 target에서 양의 advantage를 갖는 source coordination pattern은 더 강하게 사용되고, 음의 advantage를 갖는 pattern은 자동으로 약화된다.

\[
M_z > 0 \Rightarrow \alpha_z \uparrow
\]

\[
M_z < 0 \Rightarrow \alpha_z \downarrow
\]

---

## 12. Final Target Policy

Agent \(i\)의 target policy logit은 세 branch의 결합으로 정의한다.

\[
\ell^T_{i,t}
=
\ell^{\text{free}}_{i,t}
+
\ell^{\text{skill}}_{i,t}
+
\sum_z
\alpha_{z,t}
\, g_\psi(z \mid h_{1,t}, \dots, h_{N,t}, c_T)
\, \ell^S_{P_t(i),z,t}
\]

여기서

\[
\ell^{\text{free}}_{i,t} = f_{\text{free}}(h_{i,t})
\]

\[
\ell^{\text{skill}}_{i,t} = f_{\text{skill}}(h_{i,t}, u_{i,t})
\]

\[
\ell^S_{P_t(i),z,t} = f^S_{P_t(i),z}(h_{i,t})
\]

최종 action distribution은 다음과 같다.

\[
\pi_i^T(a_i \mid o_{i,t})
=
\mathrm{softmax}(\ell^T_{i,t})
\]

이 구조에서 source coordination은 target policy를 강제하지 않고, target return signal에 따라 조절되는 soft prior로만 작용한다.

---

## 13. Centralized Critic and CTDE

학습은 centralized training with decentralized execution 구조로 구성한다. Critic은 global state 또는 joint observation을 사용하여 value를 추정한다.

\[
V_\omega(s_t)
\]

Actor는 각 agent의 local observation과 필요한 latent signal을 기반으로 action을 선택한다.

\[
a_{i,t} \sim \pi_i^T(\cdot \mid o_{i,t})
\]

Centralized critic은 target advantage를 계산하여 policy update와 coordination gate update에 사용된다.

---

## 14. Training Objective

전체 학습 objective는 다음과 같다.

\[
\mathcal{L}
=
\mathcal{L}_{\text{RL}}
+
\lambda_{\text{distill}} \mathcal{L}_{\text{safe-distill}}
+
\lambda_{\text{gate}} \mathcal{L}_{\text{gate}}
+
\lambda_{\text{skill}} \mathcal{L}_{\text{skill}}
+
\lambda_{\text{coord}} \mathcal{L}_{\text{coord}}
\]

---

### 14.1 Reinforcement Learning Loss

Actor-critic 기반 objective를 사용한다.

\[
\mathcal{L}_{\text{RL}}
=
\mathcal{L}_{\text{actor}}
+
 c_v \mathcal{L}_{\text{critic}}
-
 c_H \mathcal{H}(\pi^T)
\]

Actor objective는 target advantage를 사용한다.

\[
\mathcal{L}_{\text{actor}}
=
-
\mathbb{E}_t
\left[
\log \pi_i^T(a_{i,t} \mid o_{i,t}) \widehat{A}_t
\right]
\]

Critic loss는 다음과 같다.

\[
\mathcal{L}_{\text{critic}}
=
\mathbb{E}_t
\left[
\left(V_\omega(s_t)-\widehat{R}_t\right)^2
\right]
\]

---

### 14.2 Safe Source Distillation Loss

Source coordination prior는 target에서 유용한 경우에만 distillation target으로 사용한다.

\[
\mathcal{L}_{\text{safe-distill}}
=
\sum_{i=1}^{N}
\sum_z
\lambda_z
D_{\mathrm{KL}}
\left(
\pi^S_{P_t(i),z}(\cdot \mid h_i)
\Vert
\pi_i^T(\cdot \mid o_i)
\right)
\]

여기서 distillation weight는 target usefulness score에 의해 결정된다.

\[
\lambda_z = \lambda_0 \cdot \max(0, \mathrm{Normalize}(M_z))
\]

즉 target에서 negative advantage를 보이는 coordination mode는 imitation pressure를 받지 않는다.

---

### 14.3 Coordination Gate Loss

Gate는 선택한 coordination mode가 target return을 높이면 해당 mode의 probability를 증가시키고, target return을 낮추면 probability를 감소시킨다.

\[
\mathcal{L}_{\text{gate}}
=
-
\mathbb{E}_t
\left[
\log g_\psi(z_t \mid h_{1,t}, \dots, h_{N,t}, c_T)
\widehat{A}_t
\right]
\]

Gate entropy regularization을 추가하여 특정 source pattern에 조기 고착되는 것을 방지한다.

\[
\mathcal{L}_{\text{gate-ent}}
=
-\mathcal{H}(g_\psi)
\]

---

### 14.4 Individual Skill Objective

Individual skill representation은 source trajectory와 target trajectory 모두에서 temporal consistency와 action predictability를 갖도록 학습한다.

\[
\mathcal{L}_{\text{skill}}
=
-
\mathbb{E}
\left[
\log p_\xi(a_{i,t:t+H} \mid u_{i,t}, h_{i,t})
\right]
\]

또는 option-level transition prediction objective를 사용할 수 있다.

\[
\mathcal{L}_{\text{skill-pred}}
=
\left\| \hat{h}_{i,t+H} - h_{i,t+H} \right\|^2
\]

---

### 14.5 Coordination Representation Objective

Coordination latent \(z_t\)는 joint trajectory의 temporal structure와 agent interaction pattern을 예측하도록 학습한다.

\[
\mathcal{L}_{\text{coord}}
=
-
\mathbb{E}
\left[
\log p_\phi(\mathbf{a}_{t:t+H}, \Delta s_{t:t+H} \mid z_t, s_t)
\right]
\]

또는 contrastive objective를 사용할 수 있다. 동일한 coordination mode를 공유하는 trajectory windows는 가까운 embedding을 갖고, 다른 coordination mode의 windows는 멀어지도록 학습한다.

\[
\mathcal{L}_{\text{contrastive}}
=
-
\log
\frac{
\exp(\mathrm{sim}(c_t, c_t^+)/\tau_c)
}{
\exp(\mathrm{sim}(c_t, c_t^+)/\tau_c)
+
\sum_{k}\exp(\mathrm{sim}(c_t, c_k^-)/\tau_c)
}
\]

---

## 15. Target Adaptation Algorithm

```text
Algorithm: Coordination-Aware Selective Transfer

Input:
  Source task M_S
  Target task M_T
  Source policy π^S
  Source trajectory dataset D_S
  Target interaction budget B

1. Source Training
   Train source team policy π^S on M_S.
   Collect source trajectories D_S.

2. Source Knowledge Decomposition
   Train encoder E_θ on source observations.
   Infer individual skill latents u from local trajectories.
   Infer coordination latents z from joint trajectories.
   Train source skill branch π^skill.
   Train source coordination prior heads {π^S_z}.

3. Target Initialization
   Initialize target encoder from source encoder.
   Initialize skill branch from source skill module.
   Initialize free target branch randomly or weakly from source representation.
   Initialize coordination gate with high entropy.
   Initialize transfer strength α_z for each coordination mode.

4. Online Target Learning
   For each target rollout:
     Observe local observations {o_i}.
     Compute agent embeddings h_i = E_θ(o_i).
     Estimate target context c_T from early target interaction data.
     Sample or compute role mapping P_t.
     Compute gate distribution g_ψ(z | h_1, ..., h_N, c_T).
     Select coordination mode z_t.
     Compute final policy logits using free, skill, and source prior branches.
     Sample decentralized actions {a_i}.
     Execute joint action in target task.
     Receive team reward r_t.
     Estimate target advantage Â_t using centralized critic.
     Update target actor and critic.
     Update coordination gate using Â_t.
     Update mode usefulness score M_z.
     Adjust transfer strength α_z.

5. Evaluation
   Measure target AUC, time-to-threshold, final return,
   and coordination diagnostic metrics.
```

---

## 16. Evaluation Protocol

The evaluation should distinguish three effects:

1. whether source knowledge improves target sample efficiency,
2. whether full source transfer can cause negative transfer,
3. whether selective coordination transfer reduces negative transfer while preserving useful skills.

---

### 16.1 Main Performance Metrics

#### Target Learning AUC

\[
\mathrm{AUC}(B)=\sum_{t=1}^{B} J_t
\]

여기서 \(J_t\)는 target training 중 evaluation return이다.

#### Coordination Negative Transfer Score

\[
\mathrm{CNT}(B)
=
\mathrm{AUC}_{\text{scratch}}(B)
-
\mathrm{AUC}_{\text{transfer}}(B)
\]

\[
\mathrm{CNT}(B)>0
\]

이면 해당 transfer method는 scratch보다 낮은 sample efficiency를 보인다.

#### Time-to-Threshold

Target return이 특정 threshold \(R^*\)에 도달하는 데 필요한 interaction 수를 측정한다.

\[
T_{R^*} = \min \{t : J_t \ge R^*\}
\]

#### Asymptotic Performance

충분한 interaction 이후의 final return을 비교한다.

---

### 16.2 Coordination Diagnostic Metrics

Reward metric만으로는 negative transfer의 원인을 알기 어렵다. 따라서 joint behavior의 변화를 측정하는 diagnostic metric을 함께 사용한다.

#### Role Entropy

Agent별 subtask 분포를 \(p_i(k)\)라고 할 때 role entropy는 다음과 같다.

\[
H_i^{\text{role}} = -\sum_k p_i(k) \log p_i(k)
\]

낮은 entropy는 특정 역할에 강하게 고착되었음을 의미할 수 있다.

#### Role Switching Rate

시간에 따라 agent의 inferred role이 얼마나 자주 바뀌는지 측정한다.

\[
\mathrm{RSR}
=
\frac{1}{T-1}
\sum_{t=1}^{T-1}
\mathbb{I}[\hat{r}_{i,t} \neq \hat{r}_{i,t+1}]
\]

#### Joint Action Mutual Information

Agent 간 action dependency를 측정한다.

\[
I(a_i; a_j)
=
\sum_{a_i,a_j}
 p(a_i,a_j)
 \log
 \frac{p(a_i,a_j)}{p(a_i)p(a_j)}
\]

#### Inter-Agent Distance Statistics

Agents가 함께 움직이는지, 분산되는지, 특정 agent를 따라가는지 분석하기 위해 agent 간 거리의 평균과 분산을 측정한다.

\[
d_{ij,t}=\|x_{i,t}-x_{j,t}\|
\]

#### Waiting / Inactivity Rate

Agent가 행동하지 않거나 다른 agent를 기다리는 비율을 측정한다.

\[
\mathrm{WaitRate}_i
=
\frac{1}{T}
\sum_{t=1}^{T}
\mathbb{I}[a_{i,t}=\text{inactive}]
\]

#### Joint Trajectory Coverage

Target에서 방문한 joint state distribution의 다양성을 측정한다.

\[
\mathrm{Coverage}
=
|\{s_t : s_t \in \tau^T\}|
\]

또는 source-induced visitation bias를 보기 위해 source와 target visitation distribution 사이의 divergence를 측정한다.

\[
D_{\mathrm{JS}}(d^{\pi^T}_T \Vert d^{\pi^S}_S)
\]

#### Coordination Mode Persistence

Source coordination mode가 target에서도 과도하게 유지되는지 측정한다.

\[
\mathrm{Persistence}(z)
=
\frac{1}{T}
\sum_{t=1}^{T}
\mathbb{I}[\hat{z}_t=z]
\]

---

## 17. Baselines and Ablations

제안 방법의 효과를 분해하기 위해 다음 baseline과 ablation을 사용한다.

| Method | Purpose |
|---|---|
| Scratch | Target task에서 처음부터 학습 |
| Full Policy Transfer | Source policy 전체를 target에 초기화 또는 distillation |
| Representation-Only Transfer | Encoder만 전이 |
| Skill-Only Transfer | Individual skill module만 전이 |
| Coordination-Only Transfer | Coordination prior만 전이 |
| Transfer without Gate | Source prior를 항상 사용 |
| Transfer without Free Branch | 새로운 target coordination 발견 능력 확인 |
| Transfer without Role Adapter | Agent-role mismatch 처리 능력 확인 |
| Transfer without Advantage Weighting | Harmful coordination suppress 효과 확인 |
| Selective Transfer | Full proposed method |

핵심 비교는 다음이다.

\[
\text{Full Transfer} < \text{Scratch}
\]

이면 negative transfer가 발생한 것이다.

\[
\text{Skill-Only Transfer} > \text{Full Transfer}
\]

이면 source knowledge 전체가 해로운 것이 아니라, source-specific coordination prior가 문제였음을 시사한다.

\[
\text{Selective Transfer} > \text{Skill-Only Transfer}
\]

이면 유용한 coordination prior는 유지하고 harmful coordination prior만 억제하는 것이 효과적임을 보여준다.

---

## 18. Expected Behavioral Pattern

제안 방법이 제대로 작동하면 다음 경향을 보여야 한다.

1. Source와 target의 coordination requirement가 유사한 경우, source coordination prior가 유지되어 target learning이 가속된다.
2. Source와 target의 coordination requirement가 다른 경우, harmful source pattern의 transfer strength가 감소한다.
3. Target에서 필요한 새로운 coordination pattern이 free branch를 통해 발견된다.
4. Full transfer는 source-style joint behavior를 오래 유지하지만, selective transfer는 target reward signal에 따라 joint behavior를 빠르게 재구성한다.
5. Skill-only transfer는 full transfer보다 안정적일 수 있으나, selective transfer는 유용한 coordination까지 활용하므로 더 높은 sample efficiency를 달성할 수 있다.

---

## 19. Main Hypothesis

본 방법론의 주요 가설은 다음과 같다.

### Hypothesis 1: Source coordination priors can induce negative transfer.

Target task의 coordination requirement가 source와 다르면, full source policy transfer는 scratch learning보다 낮은 early-stage learning performance를 보일 수 있다.

\[
\mathrm{AUC}_{\text{full-transfer}}(B)
<
\mathrm{AUC}_{\text{scratch}}(B)
\]

---

### Hypothesis 2: Individual skill transfer is more robust than full policy transfer.

Source에서 학습된 individual skill은 coordination prior보다 target mismatch에 덜 민감하다.

\[
\mathrm{AUC}_{\text{skill-only}}(B)
>
\mathrm{AUC}_{\text{full-transfer}}(B)
\]

---

### Hypothesis 3: Advantage-gated coordination transfer reduces negative transfer.

Target advantage에 따라 source coordination prior의 사용을 조절하면 full transfer보다 높은 sample efficiency를 얻을 수 있다.

\[
\mathrm{AUC}_{\text{selective-transfer}}(B)
>
\mathrm{AUC}_{\text{full-transfer}}(B)
\]

---

### Hypothesis 4: Free target branch is necessary for discovering new coordination.

Source coordination library에 없는 target-specific coordination이 필요한 경우, free branch가 없는 transfer method는 target exploration이 제한된다.

\[
\mathrm{AUC}_{\text{with-free-branch}}(B)
>
\mathrm{AUC}_{\text{without-free-branch}}(B)
\]

---

## 20. Contribution Summary

본 방법론의 기여는 다음과 같다.

1. Cooperative MARL transfer에서 source-specific coordination pattern이 target learning을 방해하는 coordination-induced negative transfer 문제를 정의한다.
2. Source knowledge를 representation, individual skill, role prior, coordination pattern으로 분해하는 transfer framework를 제안한다.
3. Target online advantage를 이용하여 source coordination prior의 transfer strength를 조절하는 negative-aware coordination gate를 도입한다.
4. Agent identity와 functional role의 mismatch를 완화하기 위한 role/permutation adapter를 제안한다.
5. Source-free target branch를 통해 target-specific coordination discovery를 보장한다.
6. Negative transfer를 reward-level metric과 behavior-level coordination diagnostic으로 함께 측정하는 평가 프로토콜을 제시한다.

---

## 21. Method Name

제안 방법의 이름은 다음과 같이 둘 수 있다.

> **CAST: Coordination-Aware Selective Transfer**

또는 actor-critic backbone과 결합한 형태로 다음처럼 부를 수 있다.

> **CAST-MARL**

핵심 아이디어는 다음 한 문장으로 요약된다.

> CAST reuses source skills while treating source coordination patterns as target-validated, suppressible priors rather than fixed behaviors.

한국어로는 다음과 같이 요약할 수 있다.

> CAST는 source에서 학습된 individual skill은 적극적으로 재사용하되, source coordination pattern은 target online interaction에서 검증된 경우에만 선택적으로 사용하는 cooperative MARL transfer framework이다.
