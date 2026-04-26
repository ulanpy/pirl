# PPO + HJB Architecture Graph

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

    %% ===================== HJB BRANCH IN AGENT =====================
    subgraph HJB["PPOHjbRNN training additions"]
        H1["HJB state: x = [d, psi] from raw vec"]
        H2["grad_x V via autograd on critic input"]
        H3["Hamiltonian H_r(x, u, grad V) = -l + grad V . f - rho V"]
        H4["Loss: hjb_loss_scale * E[H_r^2]"]
    end

    L --> H1
    VOUT --> H2
    H1 --> H3
    H2 --> H3
    H3 --> H4

    %% ===================== LOSSES & GRADIENT FLOW =====================
    subgraph LOSSES["PPO update losses (training)"]
        PL["L_policy (PPO clipped surrogate)"]
        EL["L_entropy"]
        VL["L_value (MSE to returns, with value clipping)"]
        HL["L_HJB (squared Bellman residual on critic)"]
        TL["L_total = PL + EL + VL + HL"]
    end

    API --> PL
    API --> EL
    VOUT --> VL
    H4 --> HL

    PL --> TL
    EL --> TL
    VL --> TL
    HL --> TL

    TL --> GP["Backprop to policy parameters"]
    TL --> GV["Backprop to value parameters"]

    GP --> NOTE1["policy grads: PPO + entropy"]
    GV --> NOTE2["value grads: critic regression + HJB residual"]
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
+ \mathcal{L}_{\text{ent}}
+ \mathcal{L}_{V}
+ \mathcal{L}_{\text{HJB}}
\]

If `hjb_loss_scale = 0`, the last term vanishes and the agent reduces to plain PPO-RNN.

### Parameter updates

Actor parameters \(\theta\) receive gradients from \(\mathcal L_\pi + \mathcal L_{\text{ent}}\):

\[
\theta \leftarrow \theta - \eta \nabla_\theta
\left(\mathcal{L}_{\pi} + \mathcal{L}_{\text{ent}}\right)
\]

Critic parameters \(\phi\) receive gradients from \(\mathcal L_V + \mathcal L_{\text{HJB}}\):

\[
\phi \leftarrow \phi - \eta \nabla_\phi
\left(\mathcal{L}_{V} + \mathcal{L}_{\text{HJB}}\right)
\]

So `separate: True` means separate parameter sets and separate gradient paths,
even though all losses are summed into one optimizer step.

### If backbone were truly shared

A shared actor-critic backbone would look like:

\[
z_t = f_\omega(s_t),\quad
\pi_\theta(a_t\mid z_t),\quad
V_\phi(z_t)
\]

Then shared backbone parameters \(\omega\) would receive combined gradients from both
policy and value losses (and from HJB through the critic head). That is not the case
in the current configuration.
