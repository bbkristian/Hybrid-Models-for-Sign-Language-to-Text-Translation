# Hybrid-Models-for-Sign-Language-to-Text-Translation
This project investigates the feasibility of translating American Sign Language (ASL) from video input into written text using a variety of deep learning models. 

## Dataset Overview

The dataset used in this work is a large-scale **Word-Level American Sign Language (WLASL)** video dataset downloaded from [Kaggle](https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed?select=nslt_2000.json), containing more than 2000 words performed by over 100 signers. It includes:

- **21,083 RGB-based videos** performed by 119 signers.
- Each video contains a single ASL sign, lasting 2.5 seconds on average.
- Detailed descriptions provided in a `WLASL_v0.3.json` file, including attributes like `gloss`, `fps`, `split`, `frame_start`, `frame_end`, `url`, and `video_id`.

## Methodology

### Transfer Learning

We utilized various pre-trained convolutional neural networks (CNNs) for video classification, adapting them for ASL recognition:

- **InceptionV3**: Captures complex spatial hierarchies.
- **ResNet50**: Learns intricate patterns without vanishing gradients.
- **EfficientNetV2L**: Balances accuracy and computational efficiency.
- **VGG16**: Strong baseline feature extraction.
- **InceptionResNetV2**: High performance with lower computational cost.

Advanced layers and mechanisms were integrated at the end of the architecture for optimization:

- **Gated Recurrent Units (GRUs)**
- **Bidirectional layers**
- **Attention mechanisms**
- **Transformers**

### Data Processing

Frames were extracted from videos and formatted for input into pre-trained CNNs. We standardized the number of frames to 90 and included only words with at least 15 videos in the training set.

### Models Performance

We evaluated models using Sparse Categorical Accuracy and TopK Categorical Accuracy (top 3 labels). Below are the results:

#### InceptionV3 Model Performance

| Configuration | Max Accuracy (%) |
|---------------|------------------|
| Transformer   | 62.07            |
| Bidirectional | 27.91            |
| Sequential    | 20.93            |
| Selfattention | 18.60            |

#### ResNet50 Model Performance

| Configuration | Max Accuracy (%) |
|---------------|------------------|
| Transformer   | 37.21            |
| Sequential    | 18.60            |
| Bidirectional | 18.60            |
| Selfattention | 16.28            |

#### EfficientNetV2L Model Performance

| Configuration | Max Accuracy (%) |
|---------------|------------------|
| Transformer   | 37.21            |
| Sequential    | 18.60            |
| Selfattention | 16.28            |
| Bidirectional | 11.63            |

#### VGG16 Model Performance

| Configuration | Max Accuracy (%) |
|---------------|------------------|
| Transformer   | 37.20            |
| Bidirectional | 13.95            |
| Selfattention | 9.30             |
| Sequential    | 11.62            |

#### InceptionResNetV2 Model Performance

| Configuration | Max Accuracy (%) |
|---------------|------------------|
| Transformer   | 34.88            |
| Bidirectional | 16.28            |
| Selfattention | 16.28            |
| Sequential    | 9.30             |

Although InceptionV3 achieved a good maximum accuracy, the accuracy curve was highly noisy and the validation loss started to diverge, indicating overfitting. This variability is likely due to the low quality of the dataset, which remains the primary challenge of our study.

![](https://github.com/bbkristian/Hybrid-Models-for-Sign-Language-to-Text-Translation/blob/main/images/inceptionv3%2Btrans_loss.png)
---------

## Landmarks CNN

In the context of computer vision and machine learning, a landmark refers to specific, predefined points on an object that are used to understand its structure and spatial configuration. Landmarks are typically chosen because they are stable, easily identifiable, and relevant.
We used Google MediaPipe to extract hand landmarks from video frames and fed them into a 2D CNN for translation tasks. 
![](https://github.com/bbkristian/Hybrid-Models-for-Sign-Language-to-Text-Translation/blob/main/images/landmarks.png)

After extracting landmarks, we give them to a CNN with the following architecture:
- 2 Convolutional Layers: They take inputs with 3 channels, 1 temporal dimension, and 1 spatial dimension. The convolution is applied only on the temporal one.
- Batch Normalization: Normalizes the output of convolutional layers. There is one after every convolution
- Max Pooling:  We perform 2 max pooling operations after every normalization. Pooling is applied only on the temporal dimension.
- ReLu: activation function after every pooling layer.
- Dropout



## Conclusive Analysis
Despite experimenting with various architectures, some models underperformed due to the low quality of the dataset. However, our best-performing model, the 2D-CNN using landmarks, achieved promising results even with an increased label set.

The robustness of our approach was evaluated on a larger set of 1000 labels, achieving a maximum accuracy of 27%. The stabilization of validation loss and accuracy indicates the model's potential for handling large-scale classification tasks within a simplified architecture.
![](https://github.com/bbkristian/Hybrid-Models-for-Sign-Language-to-Text-Translation/blob/main/images/1000_labels.png)

## References

- WLASL (World Level American Sign Language). [Kaggle Dataset](https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed?select=nslt_2000.json)
