# PPO + CBF-HJB Architecture Graph

```mermaid
flowchart TD
    O["Obs Dict: {vec, costmap}"] --> F["skrl flatten Dict (sorted keys)"]
    F --> L["obs_layout.get_vec_costmap_layout"]

    L --> A0["Actor: RecurrentGaussianPolicy"]
    A0 --> AV["vec: path, velocities, nearest obstacle, history tail"]
    A0 --> AC["costmap history"]
    AV --> AV1["vec MLP"]
    AC --> ACNN["costmap CNN"]
    AV1 --> AF["fusion MLP"]
    ACNN --> AF
    AF --> AGRU["GRU hidden state"]
    AGRU --> AM["mean action"]
    AGRU --> ALS["log_std"]
    AM --> API["Gaussian policy pi(a|h)"]
    ALS --> API

    L --> C0["Critic: FeedForwardDeterministicValue"]
    C0 --> CV["vec"]
    C0 --> CC["costmap"]
    CV --> CV1["vec MLP"]
    CC --> CCNN["costmap CNN"]
    CV1 --> CF["fusion MLP"]
    CCNN --> CF
    CF --> VH["value head"]
    VH --> VOUT["V(s)"]

    subgraph hjbBranch [CBF-HJB Regularizer]
        H1["path state: d, psi"]
        H2["nearest obstacle: x_o, y_o"]
        H3["grad V wrt d, psi"]
        H4["analytic path-HJB control"]
        H5["CBF projection on v"]
        H6["Hamiltonian residual"]
        H7["HJB loss"]
    end

    L --> H1
    L --> H2
    VOUT --> H3
    H1 --> H4
    H3 --> H4
    H2 --> H5
    H4 --> H5
    H5 --> H6
    H1 --> H6
    H3 --> H6
    H6 --> H7

    subgraph losses [Training Losses]
        PL["PPO policy loss"]
        EL["entropy loss"]
        VL["value loss"]
        HL["CBF-HJB loss"]
        TL["total loss"]
    end

    API --> PL
    API --> EL
    VOUT --> VL
    H7 --> HL
    PL --> TL
    EL --> TL
    VL --> TL
    HL --> TL
```

With `separate: True`, actor and critic have separate parameter sets. The CBF-HJB
branch regularizes the critic through `V(s)` and does not directly update the actor.
The actor is affected only indirectly through improved value targets/advantages during
PPO training.
