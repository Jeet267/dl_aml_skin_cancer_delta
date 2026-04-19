# Data Flow Architecture Diagram (Up to Phase 2)
This diagram fulfills the **Deliverable Requirements (Architecture Diagram)**, outlining the data pipeline limited to the Advanced Probabilistic ML (Phase 1) and the Deep Neural Network Integration (Phase 2).

```mermaid
graph TD
    %% Define styles
    classDef dataFill fill:#2d3748,stroke:#4a5568,stroke-width:2px,color:#fff;
    classDef mlFill fill:#2b6cb0,stroke:#3182ce,stroke-width:2px,color:#fff;
    classDef dlFill fill:#c53030,stroke:#e53e3e,stroke-width:2px,color:#fff;
    classDef metricFill fill:#d69e2e,stroke:#ecc94b,stroke-width:2px,color:#fff;

    %% Data Ingestion Nodes
    OriginalData["ISIC 2024 Raw SLICE-3D Data"]:::dataFill
    OriginalData --> TabularData["Tabular Metadata (55 cols)"]:::dataFill
    OriginalData --> ImageData["HDF5 JPEG Images"]:::dataFill

    %% Phase 1: Advanced ML
    subgraph "Phase 1: Advanced Probabilistic ML"
        TabularData --> F_Eng["Feature Engineering (Color, Z-Scores)"]
        F_Eng --> F_Sel["Feature Selection"]
        
        F_Sel --> GBDT_Ensemble["GBDT Probabilistic Ensemble"]:::mlFill
        GBDT_Ensemble --> |"Classification Decision"| OOF_Tab["Model A: Tabular Prediction Output"]:::metricFill
    end

    %% Phase 2: Deep Learning
    subgraph "Phase 2: Modern Deep Learning Architecture"
        ImageData --> Aug["Albumentations (Flips, Color Jitter)"]
        Aug --> ResNet50["ResNet50 Backbone (Pretrained)"]:::dlFill
        ResNet50 --> FocalLoss["Focal Loss (alpha=0.25, gamma=2)"]
        FocalLoss --> |"Overfitting Prevention: LR Scheduling"| Optimize["AdamW Optimization"]
        
        Optimize --> |"TTA Component"| OOF_Img["Model B: Image Prediction Output"]:::metricFill
    end
```

## Integration Concept
The diagram showcases a dual-channel analytical pipeline where structurally complex image inputs are routed through specialized **Deep Learning rigorously architected blocks (ResNet50)**, while tabular clinical data flows exclusively through **Advanced Probabilistic approaches (Ensembles)**.
