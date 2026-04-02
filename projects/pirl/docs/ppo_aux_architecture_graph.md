# PPO Aux Architecture Graph

```mermaid
flowchart TD
    O["Obs Dict: {vec, costmap}"] --> F["skrl flatten Dict (sorted keys)"]
    F --> L["obs_layout.get_vec_costmap_layout -> vec slice + costmap slice"]

    %% ===================== ACTOR =====================
    L --> A0["Actor: RecurrentGaussianPolicy"]

    A0 --> AV["vec (dim=V)"]
    A0 --> AC["costmap (C,H,W)"]

    AV --> AVS["Split vec: core(V-9) + aux(9)"]
    AVS --> AV1["core MLP: Linear->64->ELU->Linear->64->ELU"]
    AVS --> AV2["aux MLP: Linear->32->ELU->Linear->32->ELU"]

    AC --> ACNN["CNN: Conv C->16 -> Conv 16->32 -> Conv 32->64 -> Conv 64->64 -> Flatten"]

    AV1 --> AJ
    AV2 --> AJ
    ACNN --> AJ
    AJ["Concat [64 + 32 + cnn_dim]"] --> AF["Fusion: Linear->256->ELU->Linear->128->ELU"]
    AF --> ALN1["LayerNorm(128)"]
    ALN1 --> AGRU["GRU: input=128, hidden=256, layers=1, seq_len=64"]
    AGRU --> ALN2["LayerNorm(256)"]
    ALN2 --> AM["Mean head: Linear 256->2"]
    ALN2 --> ALS["log_std parameter (2,)"]

    AM --> API["Gaussian policy pi(a|h)"]
    ALS --> API

    %% ===================== CRITIC =====================
    L --> C0["Critic: FeedForwardDeterministicValue"]

    C0 --> CV["vec (dim=V)"]
    C0 --> CC["costmap (C,H,W)"]

    CV --> CV1["vec MLP: Linear->64->ELU->Linear->64->ELU"]
    CC --> CCNN["CNN: Conv C->16 -> Conv 16->32 -> Conv 32->64 -> Conv 64->64 -> Flatten"]

    CV1 --> CJ
    CCNN --> CJ
    CJ["Concat [64 + cnn_dim] (e.g., 1664)"] --> CF["Fusion: Linear->256->ELU->Linear->128->ELU"]
    CF --> VH["Value head: Linear 128->1"]
    VH --> VOUT["V(s)"]

    %% ===================== AUX LOSSES IN AGENT =====================
    subgraph AUX["PPODynamicsAuxRNN training additions"]
        D0["dynamics head enabled (scale=0.02)"]
        D1["Input: [vec_t, action_t]"]
        D2["MLP: (V+2)->128->128->5"]
        D3["Target: delta vec[:5] = vec_{t+1}[:5]-vec_t[:5]"]
        D4["Loss: MSE * dynamics_loss_scale"]
        H0["HJB loss scale = 0.0 (disabled)"]
    end

    L --> D1
    API --> D1
    D1 --> D2 --> D4
    D3 --> D4

    %% ===================== LOSSES & GRADIENT FLOW =====================
    subgraph LOSSES["PPO update losses (training)"]
        PL["L_policy (PPO clipped surrogate)"]
        EL["L_entropy"]
        VL["L_value (MSE to returns, with value clipping)"]
        DL["L_dynamics (aux MSE on vec delta)"]
        HL["L_HJB (disabled: scale=0.0)"]
        TL["L_total = PL + EL + VL + DL + HL"]
    end

    API --> PL
    API --> EL
    VOUT --> VL
    D4 --> DL

    PL --> TL
    EL --> TL
    VL --> TL
    DL --> TL
    HL --> TL

    TL --> GP["Backprop to policy parameters"]
    TL --> GV["Backprop to value parameters"]
    TL --> GD["Backprop to dynamics head parameters"]

    GP --> NOTE1["policy grads: PPO + entropy + dynamics bridge"]
    GV --> NOTE2["value grads: critic regression only"]
    GD --> NOTE3["dynamics grads: aux model + shared optimizer group"]
```

## Mathematical note: `separate: True` and gradient flow

With `separate: True`, policy and value are different networks:

\[
\pi_\theta(a_t \mid h_t), \qquad V_\phi(s_t)
\]

where \(\theta\) are actor parameters and \(\phi\) are critic parameters.
In the current config, actor is recurrent (GRU) and critic is feed-forward.
So there is no shared actor-critic backbone in this setup.

### Total loss used in one PPO update

\[
\mathcal{L}_{\text{total}}
=
\mathcal{L}_{\pi}
 \mathcal{L}_{\text{ent}}
 \mathcal{L}_{V}
 \mathcal{L}_{\text{dyn}}
 \mathcal{L}_{\text{HJB}}
\]

Current config has \(\mathcal{L}_{\text{HJB}} = 0\).

### Parameter updates

\[
\theta \leftarrow \theta - \eta \nabla_\theta
\left(
\mathcal{L}_{\pi}
 \mathcal{L}_{\text{ent}}
 \mathcal{L}_{\text{dyn}}
\right)
\]

\[
\phi \leftarrow \phi - \eta \nabla_\phi \mathcal{L}_{V}
\]

Dynamics-head parameters \(\psi\):

\[
\psi \leftarrow \psi - \eta \nabla_\psi \mathcal{L}_{\text{dyn}}
\]

### Why dynamics loss can also update actor

In code, dynamics branch uses:

\[
a_{\text{dyn}}
=
a_{\text{sampled}}
\left(\mu_\theta - \text{stopgrad}(\mu_\theta)\right)
\]

Forward value is equal to \(a_{\text{sampled}}\), but gradient flows through \(\mu_\theta\).
Therefore, \(\mathcal{L}_{\text{dyn}}\) contributes to actor gradients.

### Simple toy example

Assume at one minibatch:

\[
\mathcal{L}_{\pi}=0.8,\quad
\mathcal{L}_{\text{ent}}=-0.02,\quad
\mathcal{L}_{V}=0.5,\quad
\mathcal{L}_{\text{dyn}}=0.1
\]

\[
\mathcal{L}_{\text{total}}=1.38
\]

But gradients are not "shared equally":

- actor receives gradients from \(\mathcal{L}_{\pi}, \mathcal{L}_{\text{ent}}, \mathcal{L}_{\text{dyn}}\),
- critic receives gradients from \(\mathcal{L}_{V}\) only,
- dynamics head receives gradients from \(\mathcal{L}_{\text{dyn}}\) only.

So `separate: True` means separate parameter sets and separate gradient paths, even though all losses are summed into one optimizer step.

### If backbone were truly shared

Shared actor-critic would look like:

\[
z_t = f_\omega(s_t),\quad
\pi_\theta(a_t\mid z_t),\quad
V_\phi(z_t)
\]

Then shared backbone parameters \(\omega\) would receive combined gradients from both policy and value losses.
That is not the case in the current configuration.

