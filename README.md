# Synthetic Medical Image Generation with Diffusion Model

In the realm of medical diagnostics, particularly dermatology, the creation and utilization of extensive image datasets are paramount for the training and refinement of Artificial Intelligence (AI) driven diagnostic systems. Such systems are designed to assist healthcare professionals in quickly identifying a range of skin diseases, including malignant cancers, while simultaneously mitigating privacy concerns associated with the use of real patient data. However, the development of these datasets is frequently hindered by numerous challenges, such as significant data imbalance, variability in image quality, and stringent privacy regulations. To address these issues, our project proposes the innovative application of fine-tuning a pre-trained diffusion model to generate synthetic images. This approach not only promises to augment the available data but also to enhance the overall training process of machine learning models aimed at improving the diagnosis and treatment of skin cancer.

# Result

| Model    | Test Accuracy (%) |
|----------|-------------------|
| Baseline | 81.73             |
| Hybrid   | 84.06             |

The results of our validation efforts clearly demonstrate the efficacy of incorporating synthetic images into the training regimen of diagnostic models. We saw a notable increase of 2.33% in the final Test Accuracy, i.e. it increased from a baseline accuracy of 81.73% to 84.06%. This increase not only highlights the qualitative improvements in model robustness due to the synthetic data but also reinforces the potential utility of synthetic imagery in improving the diagnostic accuracy of medical image analysis models.
