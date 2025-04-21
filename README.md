# BasicFlowGeneration
This repository contains the implementation and evaluation program for our paper "BFGen: Basic Flow Generation for Use Cases via LLM and Relational Graph Attention Networks".

Identifying interaction scenarios between a system and its actors from the high-level requirements and forming use case basic flows is crucial in requirement refinement. Traditional manual methods often yield incomplete or inaccurate flows due to engineers’ limited domain expertise, while rule-based methods—relying on predefined parsing rules—suffer from linguistic ambiguities and domain-dependent limitations. Although large language model (LLM) approaches leverage rich domain knowledge and robust natural language processing, they are constrained by input length, generation instability, and the risk of out-of-system outputs, frequently resulting in context-unaware or irrelevant flows. To overcome these challenges, this paper proposes BFGen to generate context-compliant basic flows strictly adhering to domain constraints and requirement boundaries. BFGen employs LLMs to accurately extract domain-specific terms and interactions, and it integrates an enhanced Relational Graph Attention Network with attention preservation factors to model logical dependencies and domain constraints effectively. Empirical evaluations on 13 public and 7 industrial datasets show that BFGen outperforms leading baselines by ≈14\% (Precision), ≈7-25\% (Recall), ≈11-30\% (F1 Score), and ≈10-19\% (AUC). Furthermore, our evaluations confirmed the effectiveness of both the LLM module and the attention preservation factors, as well as the robustness of BFGen.
![The Overall Pipeline of BFGen](2_figure/pipeline.png)

## Code Structure
The code is stored in the folder `1_src`, including the BFGen algorithm, models, and the various BASELINE methods involved in RQ1.

.
├── 0_dataset  
│   ├── 0_pub_dataset  
│   │   ├── 0_origin  
│   │   │   ├── 0000 - gamma j.pdf  
│   │   │   ├── 0000 - inventory.pdf  
│   │   │   ├── 2001 - hats.pdf
│   │   │   ├── 2008 - keepass.pdf
│   │   │   ├── 2008 - peering.pdf
│   │   │   ├── 2008 - viper.doc
│   │   │   ├── 2009 - inventory 2.0.pdf
│   │   │   ├── 2009 - model manager.pdf
│   │   │   ├── SMOS
│   │   │   │   ├── SMOS1.txt
│   │   │   │   ├── SMOS10.txt
│   │   │   │   ├── SMOS11.txt
│   │   │   │   ├── SMOS12.txt
│   │   │   │   ├── SMOS13.txt
│   │   │   │   ├── SMOS14.txt
│   │   │   │   ├── SMOS15.txt
│   │   │   │   ├── SMOS16.txt
│   │   │   │   ├── SMOS17.txt
│   │   │   │   ├── SMOS18.txt
│   │   │   │   ├── SMOS19.txt
│   │   │   │   ├── SMOS2.txt
│   │   │   │   ├── SMOS20.txt
│   │   │   │   ├── SMOS21.txt
│   │   │   │   ├── SMOS22.txt
│   │   │   │   ├── SMOS23.txt
│   │   │   │   ├── SMOS24.txt
│   │   │   │   ├── SMOS25.txt
│   │   │   │   ├── SMOS26.txt
│   │   │   │   ├── SMOS27.txt
│   │   │   │   ├── SMOS28.txt
│   │   │   │   ├── SMOS29.txt
│   │   │   │   ├── SMOS3.txt
│   │   │   │   ├── SMOS30.txt
│   │   │   │   ├── SMOS31.txt
│   │   │   │   ├── SMOS32.txt
│   │   │   │   ├── SMOS33.txt
│   │   │   │   ├── SMOS34.txt
│   │   │   │   ├── SMOS35.txt
│   │   │   │   ├── SMOS36.txt
│   │   │   │   ├── SMOS37.txt
│   │   │   │   ├── SMOS38.txt
│   │   │   │   ├── SMOS39.txt
│   │   │   │   ├── SMOS4.txt
│   │   │   │   ├── SMOS40.txt
│   │   │   │   ├── SMOS41.txt
│   │   │   │   ├── SMOS42.txt
│   │   │   │   ├── SMOS43.txt
│   │   │   │   ├── SMOS44.txt
│   │   │   │   ├── SMOS45.txt
│   │   │   │   ├── SMOS46.txt
│   │   │   │   ├── SMOS47.txt
│   │   │   │   ├── SMOS48.txt
│   │   │   │   ├── SMOS49.txt
│   │   │   │   ├── SMOS5.txt
│   │   │   │   ├── SMOS50.txt
│   │   │   │   ├── SMOS51.txt
│   │   │   │   ├── SMOS52.txt
│   │   │   │   ├── SMOS53.txt
│   │   │   │   ├── SMOS54.txt
│   │   │   │   ├── SMOS55.txt
│   │   │   │   ├── SMOS56.txt
│   │   │   │   ├── SMOS57.txt
│   │   │   │   ├── SMOS58.txt
│   │   │   │   ├── SMOS59.txt
│   │   │   │   ├── SMOS6.txt
│   │   │   │   ├── SMOS60.txt
│   │   │   │   ├── SMOS61.txt
│   │   │   │   ├── SMOS62.txt
│   │   │   │   ├── SMOS63.txt
│   │   │   │   ├── SMOS64.txt
│   │   │   │   ├── SMOS65.txt
│   │   │   │   ├── SMOS66.txt
│   │   │   │   ├── SMOS67.txt
│   │   │   │   ├── SMOS7.txt
│   │   │   │   ├── SMOS8.txt
│   │   │   │   └── SMOS9.txt
│   │   │   ├── eANCI
│   │   │   │   ├── EA1.txt
│   │   │   │   ├── EA10.txt
│   │   │   │   ├── EA100.txt
│   │   │   │   ├── EA101.txt
│   │   │   │   ├── EA102.txt
│   │   │   │   ├── EA103.txt
│   │   │   │   ├── EA104.txt
│   │   │   │   ├── EA105.txt
│   │   │   │   ├── EA106.txt
│   │   │   │   ├── EA107.txt
│   │   │   │   ├── EA108.txt
│   │   │   │   ├── EA109.txt
│   │   │   │   ├── EA11.txt
│   │   │   │   ├── EA110.txt
│   │   │   │   ├── EA111.txt
│   │   │   │   ├── EA112.txt
│   │   │   │   ├── EA113.txt
│   │   │   │   ├── EA114.txt
│   │   │   │   ├── EA115.txt
│   │   │   │   ├── EA116.txt
│   │   │   │   ├── EA117.txt
│   │   │   │   ├── EA118.txt
│   │   │   │   ├── EA119.txt
│   │   │   │   ├── EA12.txt
│   │   │   │   ├── EA120.txt
│   │   │   │   ├── EA121.txt
│   │   │   │   ├── EA122.txt
│   │   │   │   ├── EA123.txt
│   │   │   │   ├── EA124.txt
│   │   │   │   ├── EA125.txt
│   │   │   │   ├── EA126.txt
│   │   │   │   ├── EA127.txt
│   │   │   │   ├── EA128.txt
│   │   │   │   ├── EA129.txt
│   │   │   │   ├── EA13.txt
│   │   │   │   ├── EA130.txt
│   │   │   │   ├── EA131.txt
│   │   │   │   ├── EA132.txt
│   │   │   │   ├── EA133.txt
│   │   │   │   ├── EA134.txt
│   │   │   │   ├── EA135.txt
│   │   │   │   ├── EA136.txt
│   │   │   │   ├── EA137.txt
│   │   │   │   ├── EA138.txt
│   │   │   │   ├── EA139.txt
│   │   │   │   ├── EA14.txt
│   │   │   │   ├── EA15.txt
│   │   │   │   ├── EA16.txt
│   │   │   │   ├── EA17.txt
│   │   │   │   ├── EA18.txt
│   │   │   │   ├── EA19.txt
│   │   │   │   ├── EA2.txt
│   │   │   │   ├── EA20.txt
│   │   │   │   ├── EA21.txt
│   │   │   │   ├── EA22.txt
│   │   │   │   ├── EA23.txt
│   │   │   │   ├── EA24.txt
│   │   │   │   ├── EA25.txt
│   │   │   │   ├── EA26.txt
│   │   │   │   ├── EA27.txt
│   │   │   │   ├── EA28.txt
│   │   │   │   ├── EA29.txt
│   │   │   │   ├── EA3.txt
│   │   │   │   ├── EA30.txt
│   │   │   │   ├── EA31.txt
│   │   │   │   ├── EA32.txt
│   │   │   │   ├── EA33.txt
│   │   │   │   ├── EA34.txt
│   │   │   │   ├── EA35.txt
│   │   │   │   ├── EA36.txt
│   │   │   │   ├── EA37.txt
│   │   │   │   ├── EA38.txt
│   │   │   │   ├── EA39.txt
│   │   │   │   ├── EA4.txt
│   │   │   │   ├── EA40.txt
│   │   │   │   ├── EA41.txt
│   │   │   │   ├── EA42.txt
│   │   │   │   ├── EA43.txt
│   │   │   │   ├── EA44.txt
│   │   │   │   ├── EA45.txt
│   │   │   │   ├── EA46.txt
│   │   │   │   ├── EA47.txt
│   │   │   │   ├── EA48.txt
│   │   │   │   ├── EA49.txt
│   │   │   │   ├── EA5.txt
│   │   │   │   ├── EA50.txt
│   │   │   │   ├── EA51.txt
│   │   │   │   ├── EA52.txt
│   │   │   │   ├── EA53.txt
│   │   │   │   ├── EA54.txt
│   │   │   │   ├── EA55.txt
│   │   │   │   ├── EA56.txt
│   │   │   │   ├── EA57.txt
│   │   │   │   ├── EA58.txt
│   │   │   │   ├── EA59.txt
│   │   │   │   ├── EA6.txt
│   │   │   │   ├── EA60.txt
│   │   │   │   ├── EA61.txt
│   │   │   │   ├── EA62.txt
│   │   │   │   ├── EA63.txt
│   │   │   │   ├── EA64.txt
│   │   │   │   ├── EA65.txt
│   │   │   │   ├── EA66.txt
│   │   │   │   ├── EA67.txt
│   │   │   │   ├── EA68.txt
│   │   │   │   ├── EA69.txt
│   │   │   │   ├── EA7.txt
│   │   │   │   ├── EA70.txt
│   │   │   │   ├── EA71.txt
│   │   │   │   ├── EA72.txt
│   │   │   │   ├── EA73.txt
│   │   │   │   ├── EA74.txt
│   │   │   │   ├── EA75.txt
│   │   │   │   ├── EA76.txt
│   │   │   │   ├── EA77.txt
│   │   │   │   ├── EA78.txt
│   │   │   │   ├── EA79.txt
│   │   │   │   ├── EA8.txt
│   │   │   │   ├── EA80.txt
│   │   │   │   ├── EA81.txt
│   │   │   │   ├── EA82.txt
│   │   │   │   ├── EA83.txt
│   │   │   │   ├── EA84.txt
│   │   │   │   ├── EA85.txt
│   │   │   │   ├── EA86.txt
│   │   │   │   ├── EA87.txt
│   │   │   │   ├── EA88.txt
│   │   │   │   ├── EA89.txt
│   │   │   │   ├── EA9.txt
│   │   │   │   ├── EA90.txt
│   │   │   │   ├── EA91.txt
│   │   │   │   ├── EA92.txt
│   │   │   │   ├── EA93.txt
│   │   │   │   ├── EA94.txt
│   │   │   │   ├── EA95.txt
│   │   │   │   ├── EA96.txt
│   │   │   │   ├── EA97.txt
│   │   │   │   ├── EA98.txt
│   │   │   │   └── EA99.txt
│   │   │   ├── eTour
│   │   │   │   ├── UC1.txt
│   │   │   │   ├── UC10.txt
│   │   │   │   ├── UC11.txt
│   │   │   │   ├── UC12.txt
│   │   │   │   ├── UC13.txt
│   │   │   │   ├── UC14.txt
│   │   │   │   ├── UC15.txt
│   │   │   │   ├── UC16.txt
│   │   │   │   ├── UC17.txt
│   │   │   │   ├── UC18.txt
│   │   │   │   ├── UC19.txt
│   │   │   │   ├── UC2.txt
│   │   │   │   ├── UC20.txt
│   │   │   │   ├── UC21.txt
│   │   │   │   ├── UC22.txt
│   │   │   │   ├── UC23.txt
│   │   │   │   ├── UC24.txt
│   │   │   │   ├── UC25.txt
│   │   │   │   ├── UC26.txt
│   │   │   │   ├── UC27.txt
│   │   │   │   ├── UC28.txt
│   │   │   │   ├── UC29.txt
│   │   │   │   ├── UC3.txt
│   │   │   │   ├── UC30.txt
│   │   │   │   ├── UC31.txt
│   │   │   │   ├── UC32.txt
│   │   │   │   ├── UC33.txt
│   │   │   │   ├── UC34.txt
│   │   │   │   ├── UC35.txt
│   │   │   │   ├── UC36.txt
│   │   │   │   ├── UC37.txt
│   │   │   │   ├── UC38.txt
│   │   │   │   ├── UC39.txt
│   │   │   │   ├── UC4.txt
│   │   │   │   ├── UC40.txt
│   │   │   │   ├── UC41.txt
│   │   │   │   ├── UC42.txt
│   │   │   │   ├── UC43.txt
│   │   │   │   ├── UC44.txt
│   │   │   │   ├── UC45.txt
│   │   │   │   ├── UC46.txt
│   │   │   │   ├── UC47.txt
│   │   │   │   ├── UC48.txt
│   │   │   │   ├── UC49.txt
│   │   │   │   ├── UC5.txt
│   │   │   │   ├── UC50.txt
│   │   │   │   ├── UC51.txt
│   │   │   │   ├── UC52.txt
│   │   │   │   ├── UC53.txt
│   │   │   │   ├── UC54.txt
│   │   │   │   ├── UC55.txt
│   │   │   │   ├── UC56.txt
│   │   │   │   ├── UC57.txt
│   │   │   │   ├── UC58.txt
│   │   │   │   ├── UC6.txt
│   │   │   │   ├── UC7.txt
│   │   │   │   ├── UC8.txt
│   │   │   │   └── UC9.txt
│   │   │   ├── easyClinic
│   │   │   │   ├── 1.txt
│   │   │   │   ├── 10.txt
│   │   │   │   ├── 11.txt
│   │   │   │   ├── 12.txt
│   │   │   │   ├── 13.txt
│   │   │   │   ├── 14.txt
│   │   │   │   ├── 15.txt
│   │   │   │   ├── 16.txt
│   │   │   │   ├── 17.txt
│   │   │   │   ├── 18.txt
│   │   │   │   ├── 19.txt
│   │   │   │   ├── 2.txt
│   │   │   │   ├── 20.txt
│   │   │   │   ├── 21.txt
│   │   │   │   ├── 22.txt
│   │   │   │   ├── 23.txt
│   │   │   │   ├── 24.txt
│   │   │   │   ├── 25.txt
│   │   │   │   ├── 26.txt
│   │   │   │   ├── 27.txt
│   │   │   │   ├── 28.txt
│   │   │   │   ├── 29.txt
│   │   │   │   ├── 3.txt
│   │   │   │   ├── 30.txt
│   │   │   │   ├── 4.txt
│   │   │   │   ├── 5.txt
│   │   │   │   ├── 6.txt
│   │   │   │   ├── 7.txt
│   │   │   │   ├── 8.txt
│   │   │   │   └── 9.txt
│   │   │   └── iTrust
│   │   │       ├── UC1.txt
│   │   │       ├── UC10.txt
│   │   │       ├── UC11.txt
│   │   │       ├── UC12.txt
│   │   │       ├── UC13.txt
│   │   │       ├── UC15.txt
│   │   │       ├── UC16.txt
│   │   │       ├── UC17.txt
│   │   │       ├── UC18.txt
│   │   │       ├── UC19.txt
│   │   │       ├── UC2.txt
│   │   │       ├── UC21.txt
│   │   │       ├── UC23.txt
│   │   │       ├── UC24.txt
│   │   │       ├── UC25.txt
│   │   │       ├── UC26.txt
│   │   │       ├── UC27.txt
│   │   │       ├── UC28.txt
│   │   │       ├── UC29.txt
│   │   │       ├── UC3.txt
│   │   │       ├── UC30.txt
│   │   │       ├── UC31.txt
│   │   │       ├── UC32.txt
│   │   │       ├── UC33.txt
│   │   │       ├── UC34.txt
│   │   │       ├── UC35.txt
│   │   │       ├── UC36.txt
│   │   │       ├── UC37.txt
│   │   │       ├── UC38.txt
│   │   │       ├── UC4.txt
│   │   │       ├── UC5.txt
│   │   │       ├── UC6.txt
│   │   │       ├── UC8.txt
│   │   │       └── UC9.txt
│   │   ├── 1_translated
│   │   │   ├── uc_SMOS_en.json
│   │   │   └── uc_eANCI_en.json
│   │   └── 2_processed
│   │       ├── uc_SMOS_en.json
│   │       ├── uc_eANCI_en.json
│   │       ├── uc_easyClinic.json
│   │       ├── uc_iTrust.json
│   │       ├── uc_keepass.json
│   │       └── uc_pub_8in1.json
│   ├── 1_industrial_dataset
│   │   ├── 0_origin
│   │   │   ├── README.md
│   │   │   └── example.txt
│   │   └── 1_processed
│   │       ├── README.md
│   │       └── example.json
│   ├── 2_experiment_dataset
│   │   ├── 1_dataset_origin
│   │   │   ├── pub_format
│   │   │   │   ├── README.md
│   │   │   │   ├── pub.json
│   │   │   │   ├── pub_only_format_uctext.json
│   │   │   │   └── pub_steps.json
│   │   │   └── pub_origin
│   │   │       ├── 2008 - keepass.xml
│   │   │       ├── uc_SMOS.json
│   │   │       ├── uc_SMOS_en.json
│   │   │       ├── uc_eANCI.json
│   │   │       ├── uc_eANCI_en.json
│   │   │       ├── uc_eTour.json
│   │   │       ├── uc_easyClinic.json
│   │   │       ├── uc_iTrust.json
│   │   │       ├── uc_keepass.json
│   │   │       └── uc_pub_8in1.json
│   │   ├── 2_dataset_origin_node
│   │   │   ├── Chatgpt_4o
│   │   │   │   ├── 1st_round
│   │   │   │   │   ├── chatgpt_8p1_ground_truth.txt
│   │   │   │   │   ├── chatgpt_NCE-T_ground_truth.txt
│   │   │   │   │   ├── chatgpt_SMOS_ground_truth.txt
│   │   │   │   │   ├── chatgpt_eANCI_ground_truth.txt
│   │   │   │   │   ├── chatgpt_easyclinic_ground_truth.txt
│   │   │   │   │   ├── chatgpt_iTrust_ground_truth.txt
│   │   │   │   │   └── chatgpt_keepass_ground_truth.txt
│   │   │   │   ├── 2nd_round
│   │   │   │   │   ├── GPT_origin_extracted_merged.txt
│   │   │   │   │   ├── NCET_ucnodes_origin.txt
│   │   │   │   │   └── pub_ucnodes_1023.txt
│   │   │   │   ├── 3rd_round
│   │   │   │   │   ├── ChatGPT_pub_ground_truth.json
│   │   │   │   │   ├── README.me
│   │   │   │   │   └── with_key
│   │   │   │   │       └── ChatGPT_pub_ground_truth.json
│   │   │   │   └── 4rd_after_formalized
│   │   │   │       ├── GPT_pub_gt.json
│   │   │   │       ├── GPT_pub_gt_withoutkey.json
│   │   │   │       └── README.md
│   │   │   ├── ERNIE-Speed-8k
│   │   │   │   ├── Ernie_8in1_groud_truth.json
│   │   │   │   ├── Ernie_NEC-T_groud_truth.json
│   │   │   │   ├── Ernie_NEC-T_groud_truth_reextract.json
│   │   │   │   ├── Ernie_NEC-T_groud_truth_with_keyword.json
│   │   │   │   ├── Ernie_NEC-T_groud_truth_with_keyword111.json
│   │   │   │   ├── Ernie_SMOS_groud_truth.json
│   │   │   │   ├── Ernie_eANCI_groud_truth.json
│   │   │   │   ├── Ernie_eTour_groud_truth.json
│   │   │   │   ├── Ernie_easyClinic_groud_truth.json
│   │   │   │   ├── Ernie_iTrust_groud_truth.json
│   │   │   │   └── Ernie_keepass_groud_truth.json
│   │   │   ├── Ernie-4-Turbo
│   │   │   │   ├── 2_pub_after_formalized
│   │   │   │   │   ├── Ernie_pub_gt.json
│   │   │   │   │   ├── README.md
│   │   │   │   │   └── ernie_pub_gt_withoutkey.json
│   │   │   │   ├── Ernie_NCET_ground_truth.json
│   │   │   │   ├── Ernie_pub_ground_truth.json
│   │   │   │   └── with_keyword
│   │   │   │       ├── Ernie_NCET_ground_truth.json
│   │   │   │       ├── Ernie_pub_ground_truth.json
│   │   │   │       └── README.md
│   │   │   └── README.md
│   │   ├── 3_dataset_prediction_step
│   │   │   ├── Chatgpt_4o
│   │   │   │   ├── 1st_round
│   │   │   │   │   ├── chatgpt_4o_NEC-T_unprocess.txt
│   │   │   │   │   ├── chatgpt_8p1_steps.txt
│   │   │   │   │   ├── chatgpt_NCE-T_steps.json
│   │   │   │   │   ├── chatgpt_SMOS_steps.txt
│   │   │   │   │   ├── chatgpt_eANCI_steps.txt
│   │   │   │   │   ├── chatgpt_easyclinic_steps.txt
│   │   │   │   │   ├── chatgpt_iTrust_steps.txt
│   │   │   │   │   └── chatgpt_keepass_steps.txt
│   │   │   │   └── 2nd_round
│   │   │   │       ├── NCET_steps_generated_by_gpt.txt
│   │   │   │       └── pub_dataset_steps_generated_by_gpt.txt
│   │   │   └── ERNIE-Speed-8k
│   │   │       ├── Ernie_8in1_steps.json
│   │   │       ├── Ernie_NEC-T_steps.json
│   │   │       ├── Ernie_SMOS_steps.json
│   │   │       ├── Ernie_eANCI_steps.json
│   │   │       ├── Ernie_eTour_steps.json
│   │   │       ├── Ernie_easyClinic_steps.json
│   │   │       ├── Ernie_iTrust_steps.json
│   │   │       └── Ernie_keepass_steps.json
│   │   ├── 4_dataset_pred_node
│   │   │   ├── ERNIE-Speed-8k
│   │   │   │   ├── Ernie_8in1.json
│   │   │   │   ├── Ernie_NEC-T.json
│   │   │   │   ├── Ernie_SMOS.json
│   │   │   │   ├── Ernie_eANCI.json
│   │   │   │   ├── Ernie_eTour.json
│   │   │   │   ├── Ernie_easyClinic.json
│   │   │   │   ├── Ernie_iTrust.json
│   │   │   │   └── Ernie_keepass.json
│   │   │   ├── ERNIE_4_Turbo_8k
│   │   │   │   └── with_tp
│   │   │   │       ├── 1st_round
│   │   │   │       │   └── Ernie_pub.json
│   │   │   │       ├── 2nd_round
│   │   │   │       │   ├── Ernie_NCE-T.json
│   │   │   │       │   ├── Ernie_pub.json
│   │   │   │       │   └── step_seg
│   │   │   │       │       ├── Ernie_pub_seg.json
│   │   │   │       │       └── README.md
│   │   │   │       └── 3rd_round
│   │   │   │           ├── Ernie_pub.json
│   │   │   │           ├── Ernie_pub_no_seg.json
│   │   │   │           └── README.md
│   │   │   └── chatgpt_4o
│   │   │       ├── 1st_round
│   │   │       │   ├── chatgpt_4o_NCE-T.json
│   │   │       │   ├── chatgpt_4o_iTrust.txt
│   │   │       │   ├── chatgpt_8p1.txt
│   │   │       │   ├── chatgpt_SMOS.txt
│   │   │       │   ├── chatgpt_eANCI.txt
│   │   │       │   ├── chatgpt_easyclinic.txt
│   │   │       │   └── chatgpt_keepass.txt
│   │   │       ├── 2nd_round
│   │   │       │   ├── GPT_generated_merged_NCET.txt
│   │   │       │   ├── GPT_generated_merged_NCET_原始.txt
│   │   │       │   ├── GPT_generated_merged_pub.txt
│   │   │       │   ├── NCET_ucnodes_gpt_generated.txt
│   │   │       │   ├── pub_ucnodes_in_gpt_steps.txt
│   │   │       │   └── with_tp
│   │   │       │       ├── ChatGPT_pub.json
│   │   │       │       ├── Chatgpt_NCE-T.json
│   │   │       │       └── README.me
│   │   │       └── 3rd_round
│   │   │           ├── ChatGPT_pub.json
│   │   │           └── README.md
│   │   ├── 5_experiment_data
│   │   │   └── 1_result
│   │   │       └── 1_baseline
│   │   │           ├── 1_only_Ernie_Speed
│   │   │           │   ├── eANCI
│   │   │           │   │   ├── AUC.json
│   │   │           │   │   ├── F1.json
│   │   │           │   │   ├── Precision.json
│   │   │           │   │   └── Recall.json
│   │   │           │   └── easyClinic
│   │   │           │       ├── AUC.json
│   │   │           │       ├── F1.json
│   │   │           │       ├── Precision.json
│   │   │           │       └── Recall.json
│   │   │           ├── 2_only_Ernie_4_Turbo
│   │   │           │   ├── 1st_round
│   │   │           │   │   ├── 0000 - inventory.json
│   │   │           │   │   ├── 2009 - inventory 2.0.json
│   │   │           │   │   ├── Ernie_all_eval.json
│   │   │           │   │   ├── Ernie_pub.json
│   │   │           │   │   ├── README.md
│   │   │           │   │   ├── SMOS.json
│   │   │           │   │   ├── eANCI.json
│   │   │           │   │   ├── eTour.json
│   │   │           │   │   ├── easyClinic.json
│   │   │           │   │   ├── gamma j.json
│   │   │           │   │   ├── hats.json
│   │   │           │   │   ├── iTrust.json
│   │   │           │   │   ├── keepass.json
│   │   │           │   │   ├── model manager.json
│   │   │           │   │   ├── pnnl.json
│   │   │           │   │   └── viper.json
│   │   │           │   ├── 2nd_round
│   │   │           │   │   ├── nce_t
│   │   │           │   │   │   ├── Ernie_NCE_T.json
│   │   │           │   │   │   ├── Ernie_all_eval.json
│   │   │           │   │   │   └── NCE-T.json
│   │   │           │   │   └── pub_dataset
│   │   │           │   │       ├── 0000 - inventory.json
│   │   │           │   │       ├── 2009 - inventory 2.0.json
│   │   │           │   │       ├── Ernie_all_eval.json
│   │   │           │   │       ├── Ernie_pub.json
│   │   │           │   │       ├── SMOS.json
│   │   │           │   │       ├── eANCI.json
│   │   │           │   │       ├── eTour.json
│   │   │           │   │       ├── easyClinic.json
│   │   │           │   │       ├── gamma j.json
│   │   │           │   │       ├── hats.json
│   │   │           │   │       ├── iTrust.json
│   │   │           │   │       ├── keepass.json
│   │   │           │   │       ├── model manager.json
│   │   │           │   │       ├── pnnl.json
│   │   │           │   │       └── viper.json
│   │   │           │   └── 3rd_round
│   │   │           │       └── pub_dataset
│   │   │           │           ├── 0000 - inventory.json
│   │   │           │           ├── 2009 - inventory 2.0.json
│   │   │           │           ├── Ernie_all_eval.json
│   │   │           │           ├── Ernie_pub.json
│   │   │           │           ├── SMOS.json
│   │   │           │           ├── eANCI.json
│   │   │           │           ├── eTour.json
│   │   │           │           ├── easyClinic.json
│   │   │           │           ├── gamma j.json
│   │   │           │           ├── hats.json
│   │   │           │           ├── iTrust.json
│   │   │           │           ├── keepass.json
│   │   │           │           ├── model manager.json
│   │   │           │           ├── pnnl.json
│   │   │           │           └── viper.json
│   │   │           ├── 3_general_method
│   │   │           │   ├── NCET
│   │   │           │   │   ├── nce-t.json
│   │   │           │   │   └── nce-t_dep.json
│   │   │           │   ├── README.md
│   │   │           │   └── pub_dataset
│   │   │           │       └── pub.json
│   │   │           └── 4_chatgpt
│   │   │               ├── ncet
│   │   │               │   ├── Chatgpt_NCE-T.json
│   │   │               │   ├── Ernie_all_eval.json
│   │   │               │   └── NCE-T.json
│   │   │               └── pub
│   │   │                   ├── 0000 - inventory.json
│   │   │                   ├── 2009 - inventory 2.0.json
│   │   │                   ├── ChatGPT_pub.json
│   │   │                   ├── Ernie_all_eval.json
│   │   │                   ├── SMOS.json
│   │   │                   ├── eANCI.json
│   │   │                   ├── eTour.json
│   │   │                   ├── easyClinic.json
│   │   │                   ├── gamma j.json
│   │   │                   ├── hats.json
│   │   │                   ├── iTrust.json
│   │   │                   ├── keepass.json
│   │   │                   ├── model manager.json
│   │   │                   ├── pnnl.json
│   │   │                   └── viper.json
│   │   └── README.md
│   └── README.md
├── 1_src
│   ├── 0_baseline_in_RQ1
│   │   ├── 0_rule-based_method
│   │   │   ├── 0_for_NCE-T_dataset.py
│   │   │   └── 1_for_pub_dataset.py
│   │   ├── 2_LLM_methods
│   │   │   ├── 0_ERNIE_for_NCET.py
│   │   │   ├── 1_ERNIE_for_pub.py
│   │   │   └── 2_ERNIE_for_pub.py
│   │   └── 3_R-GAT_with_BERT
│   │       ├── 0_for_NCET.py
│   │       └── 1_for_pub.py
│   ├── 1_BFGen
│   │   ├── 0_for_pub
│   │   │   ├── 0_SIP
│   │   │   │   ├── 0_preprocessing.py
│   │   │   │   └── 1_extract_node.py
│   │   │   ├── 1_enhanced_RGAT
│   │   │   │   ├── 0_ERNIE_pub.py
│   │   │   │   ├── 0_GPT_pub.py
│   │   │   │   ├── 1_train_on_pub.py
│   │   │   │   └── RGAT_model.py
│   │   │   └── README.md
│   │   └── 1_for_industrial
│   │       ├── 0_ERNIE_NCET.py
│   │       ├── 2_train_on_NCET.py
│   │       └── README.md
│   └── README.md
├── 2_figure
│   ├── Heterogeneous_Graph.png
│   ├── RQ2-P1.png
│   ├── RQ2-P2.png
│   ├── RQ3-lambda.png
│   ├── RQ4-P1.png
│   ├── pipeline.png
│   └── prompt.png
├── LICENSE
└── README.md


## Datasets
The datasets are stored in the folder `0_datasets`, include:
- `0_datasets/0_pub_dataset`: the public datasets.
- `0_datasets/1_industrial_dataset`: the industrial datasets.
- `0_datasets/2_experiment_dataset`  are files involved in the author's execution of the code and are related to the code. There may be partially duplication with the first two files.

## figure
The figures of the paper are stored in the folder `2_figure`.