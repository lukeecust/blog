---
title: 'Pneumonia detector: AI-assisted diagnosis from chest X-rays'
date: 2022-12-11
categories: [Projects, Public health]
description: Deep learning models for the diagnosis and classification of pneumonia.
tags: [Python,Bioinformatics,Machine learning, Computer vision, Deep learning]
pin: false
---

In this project, we developed Pneumonia Detector, a software tool designed to assist in diagnosing pneumonia, a lung infection caused by bacteria, viruses, and other pathogens. Unlike existing classifiers that merely distinguish between pneumonia and healthy lungs, our detector goes a step further --- it **simultaneously identifies the type of pneumonia** based on chest X-ray images.

By providing more detailed diagnostic insights, the software can help guide treatment decisions, enabling doctors to tailor therapies more effectively for their patients.

{% include embed/youtube.html id='q6kwzmrTaZc' %}

### Introduction
Pneumonia is an infection of the lungs that can cause inflammation and fluid filling of the air spaces in the lungs. There are approximately 450 million cases of pneumonia worldwide each year, and about 4 million people die from the disease. The main types of pneumonia are bacterial and viral pneumonia. In chest X-ray images used to diagnose pneumonia, bacterial pneumonia shows focal lobar consolidation, while more diffuse interstitial infiltrates can be seen in viral pneumonia. In contrast, a normal chest X-ray image would typically show clear, well-defined lung tissue.

Machine learning algorithms such as convolutional neural networks (CNNs) are often used for image segmentation and classification tasks on chest X-ray images. CNNs are one type of neural network designed specifically for image processing and can automatically learn features and patterns by analyzing large datasets. To reduce training time and improve model performance, transfer learning as a technique that involves using a pre-trained model on a new task was always adapted. The pre-trained model is typically trained on a large dataset and has already learned to recognize a variety of features in images. In published transfer learning models, deep CNNs such as AlexNet, VGG, ResNet, and U-Net were used to process chest X-ray images with high accuracy. Among them, the classifiers for distinguishing between normal and pneumonia have achieved high accuracy.

However, the literature focusing on multi-classification problems is significantly less than binary classification, such as classifying chest X-ray images into normal, bacterial, and viral pneumonia at one time. A multi-class classification task allows for a more detailed and specific diagnosis of the type of infection in the chest X-ray images, which can be beneficial for identifying the most appropriate treatment plan for the patients. Also, this kind of task may be a better representation of the underlying data and be more accurate in reflecting the distribution of the different types of infections. Nonetheless, a multi-class classifier is generally more complex and may be harder to train than a binary classifier.

Although there are few pieces of literature focusing on the multi-classification problem of disease-free, bacterial pneumonia, and coronavirus disease (COVID-19), the studies have limited generalizability, and may not be as sensitive to other types of viral pneumonia. The published models may only apply to periods when COVID-19 is a significant concern, and may not be as useful in situations where other types of viral pneumonia are more prevalent.

### Key Features 

- Segmentation Options: Users can choose between the U-Net model and the watershed algorithm for lung segmentation.
- Classification Models: The software supports multi-class classification, distinguishing between disease-free individuals, bacterial pneumonia, and viral pneumonia using:
	- Simple CNN
	- ResNet-50 (transfer learning)
	- DenseNet-121 (transfer learning)
- Performance:
	- Simple CNN: 86.21% accuracy
	- ResNet-50: 77.72% accuracy
	- DenseNet-121: 73.27% accuracy
	- *Training notes*: Simple CNN and DenseNet-121 may require early stopping, while ResNet-50's validation accuracy has not converged at 20 epochs and might benefit from more iterations.
- Robustness and Edge Case Handling:
	- Supports multiple image formats: JPG, PNG, TIFF, etc.
	- Prevents crashes due to missing inputs: A warning message prompts users if they attempt operations without loading an image.

### Diagnostic Insights from Segmentation:
- Normal lungs: Clear lung fields with uniformly distributed shadows.
- Bacterial pneumonia: Presence of focal lobar consolidation, particularly in the upper lobe.
- Viral pneumonia: Diffuse interstitial patterns, distinguishing it from bacterial pneumonia and normal lungs.

By integrating segmentation with classification, Pneumonia Detector enhances diagnostic accuracy and provides valuable visual cues for medical analysis.

### Source code
<https://github.com/Jiachuan-Wang/Pneumonia-Detector>

### Acknowledgements
I worked on this project with Chi Le. The presenter of the video demo is Rujie Qi. We thank Dr. Nicola Romano for his teaching in image analysis and advice on this project.