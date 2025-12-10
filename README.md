<h1>Handwritten Digit Recognition via Separable-CNN Ensembles</h1>

<hr/>

<h2>Deep Learning and Applications (UEC642)</h2>
<p>
<b>Submitted to:</b><br/>
Dr. Deepak Kumar Rakesh<br/>
Electronics and Communication Engineering Department<br/>
Thapar Institute of Engineering and Technology, Patiala<br/>
July–December 2025
</p>

<h2>Team Members</h2>
<ul>
  <li>Priyansh Saxena (102215131)</li>
  <li>Dhruv Singla (102215276)</li>
  <li>Ayush Krishna (102215085)</li>
  <li>Maulik Jain (102215063)</li>
</ul>

<h2>Abstract</h2>
<p>
This project focuses on automatic recognition of handwritten digits using the MNIST dataset, motivated by real‑world applications such as digitizing written forms, bank cheques, and postal codes. The work designs a compact convolutional neural network (CNN) based on depthwise separable convolutions to achieve high accuracy with low computational cost. On top of this, an ensemble of multiple independently trained models is employed to further boost performance. The final system attains above 99% test accuracy with strong generalization, showing that lightweight architectures combined with simple ensembling can match or exceed heavier baselines on MNIST. [web:20][web:65][web:95]
</p>

<h2>Problem Statement</h2>
<p>
Handwritten digit recognition must handle large intra‑class variation in writing styles while remaining efficient enough for deployment on resource‑constrained devices. Classical CNNs achieve strong accuracy on MNIST but often rely on over‑parameterized architectures or heavy training regimes. This project aims to design a model that is both computationally efficient and highly accurate, and to study whether model‑level techniques such as ensembles and regularization can provide measurable gains over a single CNN without changing the dataset or labels. [web:20][web:38][web:96]
</p>

<h2>Methodology</h2>

<h3>Dataset</h3>
<ul>
  <li><b>Dataset:</b> MNIST handwritten digit database (60,000 training, 10,000 test images). [web:4][web:20]</li>
  <li><b>Input format:</b> 28×28 grayscale images, single channel. [web:4]</li>
  <li><b>Preprocessing:</b> Normalization to [0,1], reshaping to (28, 28, 1), and creation of a validation split from the training set. [web:65]</li>
</ul>

<h3>Model Architecture</h3>
<ul>
  <li>
    Base model: CNN built with <b>depthwise separable convolutions</b> (SeparableConv2D) to reduce parameters and computation while preserving representational power. [web:81][web:86][web:94]
  </li>
  <li>
    Architecture:
    <ul>
      <li>Two separable‑conv blocks (32 and 64 filters) with ReLU activations, max pooling, and dropout.</li>
      <li>Fully connected head with 128 units, dropout, and a 10‑way softmax output layer.</li>
    </ul>
  </li>
  <li>
    Loss: Categorical cross‑entropy with <b>label smoothing</b> to improve generalization and calibration. [web:83][web:88]
  </li>
</ul>

<h3>Training Setup</h3>
<ul>
  <li>Framework: TensorFlow/Keras in Google Colab with GPU acceleration. [web:64][web:65]</li>
  <li>Optimizer: Adam with learning rate 1×10<sup>−3</sup>. [web:65]</li>
  <li>Batch size: 128; maximum epochs: 25 with early stopping on validation accuracy. [web:38][web:52]</li>
  <li>Learning‑rate scheduling: ReduceLROnPlateau on validation loss to refine training near convergence. [web:109]</li>
</ul>

<h3>Ensemble Strategy (Novelty)</h3>
<p>
To systematically improve accuracy beyond a single CNN, an <b>ensemble of multiple separable‑CNN models</b> is trained. Each model shares the same architecture but is initialized with a different random seed and trained independently. At inference time, softmax probability vectors from all models are averaged (soft voting), and the class with the maximum mean probability is selected as the final prediction. This approach reduces variance and provides a consistent accuracy gain over the best individual model. [web:95][web:98][web:107]
</p>

<h2>Results</h2>

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Test Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Single Separable‑CNN (label smoothing)</td>
      <td>≈ 99.0%</td>
    </tr>
    <tr>
      <td><b>Ensemble of 3 Separable‑CNNs</b></td>
      <td><b>&gt; 99.0%</b></td>
    </tr>
  </tbody>
</table>
<p>
The ensemble consistently achieves higher test accuracy than any individual base model, confirming that model diversity plus averaging reduces variance and improves generalization on MNIST. Similar ensemble strategies have been reported to yield measurable gains even when each constituent CNN is already strong. [web:95][web:38][web:104]
</p>

## Key Findings

- Depthwise separable convolutions allow construction of a compact CNN that maintains high performance on handwritten digit recognition.

- Label smoothing helps reduce overconfidence and slightly improves validation stability compared to standard one‑hot training.

- Ensembling several lightweight CNNs provides a reliable, architecture‑agnostic way to gain additional accuracy on MNIST without changing the dataset or applying heavy feature engineering.


<h2>Conclusion</h2>
<p>
The project demonstrates that efficient handwritten digit recognition can be achieved by combining modern CNN design (separable convolutions) with simple but powerful training‑time techniques (label smoothing and model ensembles). On the MNIST benchmark, this setup attains state‑of‑the‑art‑level accuracy for such a small model class while keeping training time short on a standard Colab GPU. Future extensions could explore knowledge distillation to compress the ensemble into a single student network, or adaptation of the same architecture to more challenging handwriting datasets such as EMNIST or real‑world digit corpora. [web:20][web:65][web:95]
</p>

## References

1. Y. LeCun, C. Cortes, and C.J.C. Burges, “The MNIST Database of Handwritten Digits,” http://yann.lecun.com/exdb/mnist/  
2. TensorFlow, “Simple MNIST ConvNet (Keras Example),” https://keras.io/examples/vision/mnist_convnet/  
3. C. Deotte, “How to Choose CNN Architecture MNIST,” Kaggle Notebook, https://www.kaggle.com/code/cdeotte/how-to-choose-cnn-architecture-mnist  
4. “Using Depthwise Separable Convolutions in TensorFlow,” Machine Learning Mastery, https://machinelearningmastery.com/using-depthwise-separable-convolutions-in-tensorflow/  
5. “Label Smoothing: The Overlooked and Lesser-Talked Regularization Technique,” Daily Dose of Data Science, https://blog.dailydoseofds.com/p/label-smoothing-the-overlooked-and  

