# One for Dozens: Adaptive REcommendation for All Domains with Counterfactual Augmentation (AREAD)

This is the official implementation of the AREAD model in our paper: **One for Dozens: Adaptive REcommendation for All Domains with Counterfactual Augmentation**, which has been accepted at AAAI 2025. [paper](https://arxiv.org/abs/2412.11905)

If this repository or our paper is beneficial for your work, please cite:

```
Luo, Huishi et al. “One for Dozens: Adaptive REcommendation for All Domains with Counterfactual Augmentation.” (2024).
```

or in bibtex style:

```
@misc{luo2024aread,
      title={One for Dozens: Adaptive REcommendation for All Domains with Counterfactual Augmentation}, 
      author={Huishi Luo and Yiwen Chen and Yiqing Wu and Fuzhen Zhuang and Deqing Wang},
      year={2024},
      eprint={2412.11905},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2412.11905}, 
}
```

## Introduction

Multi-domain recommendation (MDR) aims to enhance recommendation performance across various domains. However, real-world recommender systems in online platforms often need to handle dozens or even hundreds of domains, far exceeding the capabilities of traditional MDR algorithms, which typically focus on fewer than five domains. Key challenges include a substantial increase in parameter count, high maintenance costs, and intricate knowledge transfer patterns across domains. Furthermore, minor domains often suffer from data sparsity, leading to inadequate training in classical methods.

To address these issues, we propose Adaptive REcommendation for All Domains with counterfactual augmentation (AREAD). AREAD employs a hierarchical expert structure (HEI) and learns domain-specific expert network selection masks (HEMP) with counterfactual augmentation. Our experiments on two public datasets, each encompassing over twenty domains, demonstrate AREAD's effectiveness, especially in data-sparse domains.

<div align="center">
    <figure>
        <img src="./img/hei.png" width="80%" alt="Hierarchical Expert Integration (HEI)">
        <figcaption>Figure 1: Hierarchical Expert Integration (HEI)</figcaption>
    </figure>
</div>

<div align="center">
    <figure>
        <img src="./img/hemp.png" width="80%" alt="Hierarchical Expert Mask Pruning (HEMP)">
        <figcaption>Figure 2: Hierarchical Expert Mask Pruning (HEMP)</figcaption>
    </figure>
</div>


<div align="center">
    <figure>
        <img src="./img/aug1.png" width="80%" alt="Popularity-based Counterfactual Augmenter">
        <figcaption>Figure 3: Popularity-based Counterfactual Augmenter</figcaption>
    </figure>
</div>


## Datasets

For our experiments, we utilize two widely recognized datasets: the **Amazon dataset** and the **AliCCP dataset**. Both datasets are structured around item categories, serving as distinct domains for our recommendation tasks. Dataset statistics are detailed in the following table. Sample data files are included in the repository folders (`dataset/aliccp` and `dataset/amazon`) to facilitate initial setup and verification. To perform full-scale training and harness the complete capabilities of AREAD, users are required to download the complete datasets from the provided links.

**Table: Statistics of the datasets.** "Majority ratio" refers to the sample ratio in the largest domain, and "Minor domains" represent domains with less than 2% of the samples.

| Metrics            | Amazon     | AliCCP   |
|--------------------|------------|----------|
| **#Users**         | 2,899,335  | 538,358  | 
| **#Items**         | 1,368,287  | 24,096   |
| **#Samples**       | 17,664,862 | 4,454,814|
| **#Positive samples** | 8,790,575 | 329,416  |
| **#Domains**       | 25         | 30       |
| **Majority ratio** | 17%        | 60.51%   |
| **#Minor domains** | 12         | 23       |
| **Augmentation ratio** | 10%     | 10%      |


### Amazon Review Dataset

- **Download**: [Amazon Review Data (2018)](https://nijianmo.github.io/amazon/index.html).
- **Reference**: Ni, J., Li, J., McAuley, J. (2019). Justifying recommendations using distantly-labeled reviews and fine-grained aspects. In *Proceedings of EMNLP-IJCNLP 2019*, pages 188-197. 

### AliCCP Dataset

- **Download**: [AliCCP Dataset](https://tianchi.aliyun.com/dataset/408).
- **Reference**: Ma, X., Zhao, L., Huang, G., et al. (2018). Entire space multi-task model: An effective approach for estimating post-click conversion rate. In *SIGIR 2018*, pages 1137-1140. 

### Dataset Handling
The datasets selected, Amazon and AliCCP, are prepared and utilized as follows:

**Amazon Dataset:**
- We process all 25 available item categories from the dataset, covering data from the most recent 12 months.
- Users and items are filtered to include those with at least 3 interactions.
- Ratings above a user's average are labeled positive. (Labeling based on a global average would yield a positive ratio as high as 68%, which deviates from practical usage in recommendation systems.)
- We track historical sequences of positive and negative interactions from the past 6 months to enrich feature sets.
- Data is split temporally into training, validation, and test sets in a 90:5:5 ratio.

**AliCCP Dataset:**
- We focus on users and items with more than 15 interactions, sampling 30 distinct item categories as separate domains.
- This dataset employs binary labels derived from click events.
- The dataset's original training segment is used for model training, while the test set is split into validation and test subsets in equal proportions.

### Cloud-Theme Dataset

For those interested in testing recommendation algorithms across large-scale domains, the Cloud-Theme dataset may be a useful resource. This dataset includes over 1.4 million clicks across 355 distinct scenarios during a 6-day promotional period, supplemented by one month's purchase history of users before the promotion started. The label `clk_cnt` represents the number of times a user clicked on an item in the current session. Adapting this dataset for a common CTR task may require additional negative sampling techniques.

During the rebuttal phase of our paper, in response to reviewer comments, we conducted small-scale experiments using baseline models and our proposed AREAD on the Cloud-Theme dataset. Consequently, our repository includes preprocessing scripts for this dataset. We chose not to publish the experimental results as we have not yet performed comprehensive preprocessing and hyperparameter tuning on the dataset. Publishing these preliminary results without rigorous validation could compromise the fairness and integrity of our results.

- **Download**: [Cloud-Theme Dataset](https://tianchi.aliyun.com/dataset/9716).
- **Reference**:
  - Du, Z., Wang, X., Yang, H., Zhou, J., & Tang, J. (2019, July). Sequential scenario-specific meta learner for online recommendation. In *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining* (pp. 2895-2904).
  - Li, W., Zhou, J., Luo, C., Tang, C., Zhang, K., & Zhao, S. (2024, October). Scene-wise Adaptive Network for Dynamic Cold-start Scenes Optimization in CTR Prediction. In *Proceedings of the 18th ACM Conference on Recommender Systems* (pp. 370-379).



## Run
You can run this model through:
```python
# Run directly with default parameters 
python main.py

# Run the model with a different dataset, specify the `dataset_name` parameter. 
python main.py --dataset_name amazon

# Run other models, specify the `model` parameter.
python main.py --model_name aread
```


## Related Repositories
We extend our gratitude to the following repositories, whose valuable resources have been instrumental in shaping our baseline and data processing code:

1. [Torch-RecHub](https://github.com/datawhalechina/torch-rechub): Comprehensive repository for recommendation systems, offering foundational code modules and dataset processing scripts.

2. [DeepCTR-Torch](https://github.com/shenweichen/DeepCTR-Torch): PyTorch implementation of [DeepCTR](https://github.com/shenweichen/DeepCTR), featuring an easy-to-use, modular, and extendable package of deep-learning based CTR models, including core components and multi-task learning algorithms. 

3. [FuxiCTR](https://github.com/reczoo/FuxiCTR): Offers a rich collection of models for CTR prediction, behavior sequence modeling, dynamic weight networks, and multi-task learning.

4. [SATrans](https://github.com/qwerfdsaplking/SATrans): Source code for paper "Scenario-Adaptive Feature Interaction for Click-Through Rate Prediction" (KDD 2023), offering useful single-domain and multi-domain/task models.

5. [HiNet](https://github.com/mrchor/HiNet): Source code for paper "HiNet: Novel Multi-Scenario & Multi-Task Learning with Hierarchical Information Extraction" (ICDE 2023), used as a baseline in our experiments.


