# Synthetic Medical Image Generation with Diffusion Model

In the realm of medical diagnostics, particularly dermatology, the creation and utilization of extensive image datasets are paramount for the training and refinement of Artificial Intelligence (AI) driven diagnostic systems. Such systems are designed to assist healthcare professionals in quickly identifying a range of skin diseases, including malignant cancers, while simultaneously mitigating privacy concerns associated with the use of real patient data. However, the development of these datasets is frequently hindered by numerous challenges, such as significant data imbalance, variability in image quality, and stringent privacy regulations. To address these issues, our project proposes the innovative application of fine-tuning a pre-trained diffusion model to generate synthetic images. This approach not only promises to augment the available data but also to enhance the overall training process of machine learning models aimed at improving the diagnosis and treatment of skin cancer.

# Methodology
We attempted to fine-tune a diffusion model (runwayml/stable-diffusion-v1-5) using the ISIC 2020 dataset, employing DreamBooth, a targeted training methodology designed to update diffusion models by training on a minimal set of images representing specific subjects or styles. This approach involves associating a unique prompt with representative images. [https://huggingface.co/docs/diffusers/en/training/dreambooth]

In 'Derm-T2IM' paper, by Muhammad Ali Farooq et al., they developed a stable diffusion model for generating synthetic images of skin cancer which we tried to replicate. [https://arxiv.org/pdf/2401.05159]

For the validation of synthetic data, we engaged in a comprehensive evaluation of the generated images sourced from the ”Derm-T2IM” study by Muhammad Ali Farooq et al.
Our approach involved employing a skin lesion classifier based on the Vision Transformer (ViT) architecture to assess the authenticity and clinical relevance of these synthetic images. 

# Result

| Model    | Test Accuracy (%) |
|----------|-------------------|
| Baseline | 81.73             |
| Hybrid   | 84.06             |

The results of our validation efforts clearly demonstrate the efficacy of incorporating synthetic images into the training regimen of diagnostic models. We saw a notable increase of 2.33% in the final Test Accuracy, i.e. it increased from a baseline accuracy of 81.73% to 84.06%. This increase not only highlights the qualitative improvements in model robustness due to the synthetic data but also reinforces the potential utility of synthetic imagery in improving the diagnostic accuracy of medical image analysis models.
