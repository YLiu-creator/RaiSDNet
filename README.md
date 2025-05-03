# An Integrated 2D&3D Isomorphic Imaging and Cross-Modal Network for Rail Surface Defect Detection and Measurement![image](https://github.com/user-attachments/assets/cc3f0ed8-6bb2-4ba3-be56-befeee479ee0)


## Abstract
Rail defects significantly impact train operations, even posing serious safety risks. Existing methods can automatically collect images from the rail surface and identify apparent defects while facing challenges such as high false positive rates, visually subtle defects omit errors, and quantitative defect size measurement. To address these issues, this study proposes an integrated 2D&3D rail surface defect detection and measurement framework. Specifically, this framework introduces an isomorphic imaging system with a long-short exposure mechanism, which uses a single camera to simultaneously capture pixel-level registered 2D texture and 3D depth images of rail surface. Subquently, a cross-modal defect detection network is proposed to explore complementary semantic and structural information from 2D and 3D images hierarchically, enhancing defect identification capability. Finally, a partition projection-based 3D measurement method is established, considering the physical curvature changes of the railhead, providing accurate quantitative measurements for defect depth, width, and length. This study collects 2045 operational rail surface images with visible defects and establishes a standard dataset to validate model performance. Experimental results show that this technology achieves improvements of 7.26% and 9.17% in maximum F1-Score and Recall, compared to prevalent SAINet. The defect depth measurement accuracy reached 0.18 millimeters. Experiments on publicly available non-service rail surface defect datasets also demonstrate the effectiveness of the proposed method.


Here is a short implementation since the paper is under review. Further comprehensive details will be made available upon formal acceptance of the research paper.


## Dataset
Operational rail surface defect dataset: https://pan.baidu.com/s/14t9utRUWm9U30E6U5TvBkQ?pwd=hbe1 (pwd: hbe1)


## Framwork
RaiSDNet follows a U-shape architecture, containing two individual encoding branches for texture and depth feature extraction and a decoder for spatial restoration and defect detection. 
The texture encoding branch utilizes three lightweight Swin Transformer layers to extract visual semantics from texture images. 
The depth branch has the same network structure as the texture branch but with fewer parameters to retain fine-grained structure variations. 
Additionally, RaiSDNet employs convolutional layers as the neck block to generate anomaly scores, ensuring the stability of the encoding branches training. 
For cross-modal complementary feature learning, RaiSDNet devises a coarse-to-fine contrastive module (C2F) to hierarchically explore differentiated representation between 2D texture and 3D depth images. 
The fused features are sent into the Swin decoder to gradually recover spatial resolution. 
The network architecture of the Swin decoder is fully symmetric to the texture branch, and a convolutional layer is applied to predict the defect detection map.

![3overallarchirecture](https://github.com/user-attachments/assets/b618e56c-f956-463e-94ec-e0e13527b4da)










