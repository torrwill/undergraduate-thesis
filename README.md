# Will Torres' Undergraduate Thesis Repository
A repository for the code written for my thesis (paper forthcoming).

## Methods (draft)
*Based on Seth Adams' Audio Classification github repository.*[^1]
This section describes the methods used to develop the classifier:

**Download and Extract Data**
- Load audio (.wav)

**Prepare Data**
- Downsample to 16000 Mono
- Resize, if needed (define fixed length audio samples i.e., 1 second)

**Preprocess Data**
- Create a Signal Envelope to tracks the audio signal and remove dead space
- Covert to Mel Spectrogram (mel-spec)
    - the Keras Kapre[^2] library performs mel-spec in real-time using the GPU
- Spectrogram Augmentation (Time and Frequency Mask)
    - Split waves
    - Test wave threshold
    - Save samples to directory
        - If none exists, create directory

**Build Model**

This sections hosts the documentation of the following models based in TensorFlow, Keras, and SKLearn (*models are based on Evan Radkoff's Drum Sound Classification repository*[^3]):
- Random Forest (RF)
- K-Nearest Neighbors (KNN)
- Logistic Regression (LogReg)
- Support Vector Machine (SVM)
- Gradient Boosting (GB)
- Convolutional Neural Network (CNN)
- CNN + SVM

**Train Model**

This section focuses on scanning the directory and preparing a list of all the audio file paths, extracting the class label from each file name or from the name of the parent sub-folder, and mapping each class name from text to a numerical class ID.

**Perform Inference**

This section will describe the usage of this model against previously unseen data.

[^1]: [Audio Classification Repository](https://github.com/seth814/Audio-Classification/)
[^2]: [Kapre Documentation](https://github.com/keunwoochoi/kapre/)
[^3]: [Evan Radkoff Repository](https://github.com/radkoff/drum_sound_classifier)