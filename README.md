# BasicFlowGeneration
This repository contains the implementation and evaluation program for our paper "BFGen: Basic Flow Generation for Use Cases via LLM and Relational Graph Attention Networks".

Identifying interaction scenarios between a system and its actors from the high-level requirements and forming use case basic flows is crucial in requirement refinement. Traditional manual methods often yield incomplete or inaccurate flows due to engineers’ limited domain expertise, while rule-based methods—relying on predefined parsing rules—suffer from linguistic ambiguities and domain-dependent limitations. Although large language model (LLM) approaches leverage rich domain knowledge and robust natural language processing, they are constrained by input length, generation instability, and the risk of out-of-system outputs, frequently resulting in context-unaware or irrelevant flows. To overcome these challenges, this paper proposes BFGen to generate context-compliant basic flows strictly adhering to domain constraints and requirement boundaries. BFGen employs LLMs to accurately extract domain-specific terms and interactions, and it integrates an enhanced Relational Graph Attention Network with attention preservation factors to model logical dependencies and domain constraints effectively. Empirical evaluations on 13 public and 7 industrial datasets show that BFGen outperforms leading baselines by ≈14\% (Precision), ≈7-25\% (Recall), ≈11-30\% (F1 Score), and ≈10-19\% (AUC). Furthermore, our evaluations confirmed the effectiveness of both the LLM module and the attention preservation factors, as well as the robustness of BFGen.
![The Overall Pipeline of BFGen](2_figure/pipeline.png)

## Code Structure
The code is stored in the folder `src`, including the BFGen algorithm, models, and the various BASELINE methods involved in RQ1.

## Datasets
The datasets are stored in the folder `datasets`, include:
- `datasets/public`: the public datasets.
- `datasets/industrial`: the industrial datasets.

## figure
The figure of the paper is stored in the folder `figure`.