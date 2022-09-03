## Paths to a Less Artificial Intelligence

![Image](https://1.bp.blogspot.com/-CjdolPLTKxo/XzdyjCPbRwI/AAAAAAAAB0E/twI3Q6CRup8DZMQPC7wLSTsKQk36Z8KhQCLcBGAsYHQ/s850/Similarity-between-biological-and-artificial-neural-networks-Arbib-2003a-Haykin.png)

_Comparison between a biological neuron and the McCulloch-Pitts neuron (Arbib, 2003a)._

## Introduction

The human brain serves as the ultimate inspiration for the field of artificial intelligence. Of the many machine learning algorithms that have adopted properties of our nervous system to achieve its efficiency, the most renowned is the artificial neural network (ANN). Inspired by the depth and connectivity of biological neural networks, ANNs use layers of computational neurons to perform a distinct form of machine learning often called "deep learning". This technique allows them to learn from large sets of data to predict and classify information with great accuracy. The power of this technique often exceeds that of other machine learning algorithms as evidenced in a wealth of literature and its preferential use by state-of-the-art artificial intelligence systems such as Apple's Siri (Hoy, 2018). This distinction has led me to believe that the brain-inspired approach of neural networks has conferred onto the algorithm a human-like ability to recognize patterns and that biological plausibility may be a guide to further advances in artificial intelligence.
 	
Given the brain embodies many of the functions we wish to see in deep learning (e.g., power conservation and robust generalization), it is reasonable to believe that the principles of neuroscience can serve as a guide in the development of more robust deep learning and intelligent systems. Despite this, many of today's artificial neural networks are still vague representations of the biological neural networks they were originally modeled after decades ago, and the differences between the two are increasingly outweighing the similarities. This growing deviation of artificial neural networks from biological neural networks is further exacerbated by a less than optimal level of collaboration between the fields of neuroscience and computer science. To confront these challenges, the present thesis investigates how neuroscience can serve as a source of inspiration for artificial intelligence through biologically-plausible neural networks and the applications thereof.

Within this investigation of ways in which biological plausibility may improve neural networks, an existing algorithm known as a spiking neural network (SNN) will be discussed in relatively greater depth and comprise the majority of my engagement with the various routes to biological plausibility in deep learning. SNNs are computational neural networks that more closely mimic biological neural networks in that their neurons send information on their own schedule relative to others, particularly when their computational equivalent of a membrane reaches a specific threshold. Given the progress made by SNNs in achieving a biologically plausible architecture, they serve as a great foundation for a discussion on what biological plausibility in neural networks may look like and what direction should be trekked to improve upon what currently works. 

One limitation of neural networks with greater degrees of biological realism is that they are computationally expensive to simulate in traditional von-Neumann CPUs. For this reason, the pursuit of biological plausibility will be limited by the constraints of in-silico development. While this represents a major obstacle to the present goal, it is unlikely that the neural networks to be used will reach a complexity that will push the limits of the standard GPU or CPU (Hazan et al., 2018).

## Spiking Neural Networks

Spiking neural networks differ from today's conventional deep neural networks in that they incorporate the temporal dimension of neural computation. That is, instead of forwardly propagating information synchronously as is typically done by default in an ANN, the neurons of an SSN transmit info only when their computational equivalent of a membrane potential— the membrane's electrical charge— reaches a threshold. Once this threshold is achieved, this neuron "spikes" (typically represented as a zero or one in an activation function) and sends a signal to other neurons which in turn influences their membrane potential. Depending on the implementation of the SSN, after this, the value of the neuron can briefly drop below its initial activation to mimic a biological neuron’s refractory period after an action potential. In operation, multiple spikes at discrete points in time create the appearance of “spike trains” which are passed into a network and produced as output. Depending on its implementation, whether the neuron fires or does not receive the sufficient number of spikes to do so, the value of the neuron will gradually return to its baseline ([Ponulak & Kasiński, 2010](https://pubmed.ncbi.nlm.nih.gov/22237491/)). 

Over the years, various models have been developed to model the neurons of an SNN. The most popular and the one to be used here is the Leaky Integrate and Fire (LIF) threshold model which simply proposes setting the value in a neuron to a baseline activation level and then having incoming spike trains increase or decrease this value until the threshold is reached or the state decays. The main goal in this model is to train the synapses (the weights) to cause the postsynaptic neuron to generate a spike train with desired spike times (assuming latency coded input).

A code example of a leaky integrate-and-fire neural network is shown below (syntax provided by SNNTorch). Note, snn.Leaky() instantiates a simple leaky integrate-and-fire neuron.

```
import torch, torch.nn as nn
import snntorch as snn
from snntorch import surrogate

num_steps = 25 # number of time steps
batch_size = 1
beta = 0.5  # neuron decay rate
spike_grad = surrogate.fast_sigmoid()

net = nn.Sequential(
      nn.Conv2d(1, 8, 5),
      nn.MaxPool2d(2),
      snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad),
      nn.Conv2d(8, 16, 5),
      nn.MaxPool2d(2),
      snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad),
      nn.Flatten(),
      nn.Linear(16 * 4 * 4, 10),
      snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad, output=True)
      )

# random input data
data_in = torch.rand(num_steps, batch_size, 1, 28, 28)

spike_recording = []

for step in range(num_steps):
    spike, state = net(data_in[step])
    spike_recording.append(spike)

```

## Methods
Developing a neural network that is as biologically plausible as an SNN trained via local learning rules but as powerful as a deep artificial neural network delineates an important challenge in the field of deep learning. The current literature shows that, with current training techniques and hardware, there exist tradeoffs between biological plausibility and performance (Tavanei et al., 2019). My project goal is to construct SNNs whose performance and biological plausibility are not strongly inversely correlated and perform as well as, if not better than, ANNs on benchmark tasks. In addition, I hope to address the limitations of the aforementioned SNNs to further investigate the extent to which biological plausibility can benefit neural network architectures. This will entail comparing the performance of SNNs high in biological plausibility with their traditional ANN counterparts on benchmark tasks. I hypothesize that the spiking networks will perform at least comparable to ANNs per prior literature (Comsa et al., 2018).

These benchmarks include:
1. MNIST (Lecun et al., 1998)
2. CIFAR-10 (Krizhevsky, 2009)
3. Google Speech Commands (Warden, 2018)

### MNIST
#### Dataset
The first experiment compared the average performance of an ANN and SNN on the MNIST dataset. MNIST is commonly used for the evaluation of machine learning models and was appropriate as the first benchmark task. This dataset consists of 70,000 28 by 28 pixel gray-scale images of handwritten digits. 60,000 of these images were used as training samples, while 10,000 were used as testing samples.

The objective of this task is to create a model that can accurately classify an image as the number it displays. Convolutional neural networks are known to perform well on image-based datasets due to their feature extraction properties, so two were trained to complete this task-- one with rate encoded spikes (CSNN) and one without (CNN).

While training the CNN on MNIST was straightforward, training the CSNN required that the MNIST dataset first be converted to spikes since SNNs are made to exploit time-varying data and MNIST is not time-varying. To do this, I employed a data-to-spike conversion known as rate coding. Simply, rate coding uses input features to determine spiking frequency. In a rate-coded MNIST task, the probability of spiking corresponds to a pixel's brightness value: a white pixel corresponds to a 100% probability of spiking, while a black pixel corresponds to 0%. The rate coding data to spike conversion was not the preferred conversion method for this application as it does not fully exploit the temporal dynamics of SNNs and lacks the biological plausibility of latency (or temporal) coding. Granted, rate coding does have a neuroscientific relevance, it only explains a very small percentage of neurons in the primary visual cortex (Olshausen and Field, 2004) and consumes nearly as much power as ANNs.

Latency coding, on the other hand, encodes input data in the timing of single spikes. These single spikes carry as much meaning in their timing as the frequency of rate codes and consume relatively small amounts of energy. For this reason, latency coding has a slightly higher degree of biological plausibility than rate coding. Latency coding was not used here as its implementation was not the default in snnTorch. This setback outlines a future direction for this project.

Despite rate coding's relatively lower biological relevance, its use here doesn't entirely contradict the goal of creating a viable, biologically plausible SNN. Rate coding is at least consistent with the principle that "neurons that fire together, wire together" (Hebb, 1950), for plenty of spiking suggests that there is plenty of learning. Guo et al. elaborate on the biological plausibility of rate-coded SNNs, in their comprehensive review of neural coding (Guo et al., 2021).

#### Models
To ensure an accurate and fair comparison, both neural networks had the same general architecture and hyperparameters. Each network comprised an input layer of 784 neurons (one neuron per pixel) and 10 output neurons (one neuron for digits 0 to 9) and had the following sequential architecture: 8C5-MP2-16C5-MP2-1024FC10. (8C5 is a 5 x 5 convolutional kernel with 12 filters, MP2 is a 2 by 2 max-pooling function and 1024FC10 is a fully connected layer that maps 1024 neurons to 10 outputs.) In terms of differences between the two models, the most important was the use of the leaky integrate and fire (LIF) Neuron Model in the CSNN. This model, and snnTorch's implementation of it, allowed the instantiation of hidden neurons which integrated rate encoded inputs over time and emitted spikes if a certain threshold was met. Within this rate coding scheme, the output neuron with the highest firing rate (or spike count) is considered the predicted class. The goal in training, therefore, was to iteratively adjust their spike frequency such that, at the end of the training, the output neuron with the highest rate corresponds to the digit in the image (i.e. the target). This rate paradigm contrasts with conventional, non-spiking neural networks wherein the output neurons with the highest activations are treated as the predicted classes.

As LIF neurons are essentially spiking equivalents of activation functions, to create the ANN, the LIF neurons in the SNN were simply replaced with standard (i.e. non-spiking) ReLU activation functions.
      
#### Network Training
To overcome the training difficulties associated with spiking neural networks, I used snnTorch's implementation of Surrogate Gradient Descent (SGD) to train the CSNN. SGD is an increasingly popular method that addresses the nondifferentiable nonlinearity of spikes in SNNs by discovering the required connectivity to facilitate training. The choice of SGD was natural as it is a powerful and fairly biologically plausible means of training.
In terms of pipeline specifications, the SNN's rate-encoded data were fed to the model in 25 time steps. This model was trained for 10 epochs with an Adam optimizer and an MSE Count loss which tallies the number of output spikes at the end of a simulation. The optimizer used a learning rate of 0.002 and a neuron decay rate (beta) of 0.9. Similarly, in the ANN, the model was trained for 10 epochs with an SGD optimizer and cross-entropy loss. The optimizer used a learning rate of 0.01.
 
Lastly, to achieve stable and precise performance metrics for my neural networks, I trained both neural networks multiple times from scratch and averaged the metrics from each replicate. Three replicates of each network were averaged in this experiment. 

### CIFAR-10
#### Dataset
This dataset consists of 60,000 32 by 32 RGB images in 10 classes, with 6000 images per class. These classes are as follows: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. By convention, I used 50000 images to train the neural networks and 10,000 images to test them.

#### Models
In this experiment, I considered using variations of pre-existing neural networks such as VGG (Simonyan and Zissermnan, 2015) and AlexNet (Krizhevsky et al., 2012) with the reasoning that it would simplify the development of the spiking neural network and allow for a more contextualized comparison given the wealth of literature on VGGs and AlexNet. However, these networks performed significantly worse in the form of an SNN which ultimately resulted in their removal from consideration.
In lieu of a VGG and AlexNet, I used a standard convolutional neural network. Each network comprised an input layer of 1024 neurons (one neuron per pixel) and 10 output neurons (one for each class) and had the following sequential architecture: 12C3-MP2-64C3-MP2-2304FC512-512FC128-128FC10. As in the previous MNIST task, the SNN version of the CNN used LIF neurons instead of ReLU activation functions. Similarly, it was also trained via SGD.

#### Network Training
As with the MNIST task, the neural networks were trained via surrogate gradient descent.
In the SNN, rate encoded data were fed to the model in 25 time steps. This model was trained for 10 epochs with an Adam optimizer and MSE Count loss. The optimizer used a learning rate of 0.002 and a neuron decay rate (beta) of 0.9. In the ANN, the model was trained for 10 epochs with an SGD optimizer and cross-entropy loss. The optimizer used a learning rate of 0.01. Lastly, the accuracy metric reported for each neural network was the average of 3 full runs through its respective pipeline.

### Google Speech Commands
#### Dataset
The Speech Commands dataset was Google's response to many requests for an open-source dataset that can be used to develop simple speech recognition tasks. Its primary goal is, thus, to provide a standard way to build and test small models that detect when a single word is spoken from background noises and/or unrelated speech. The dataset has 65,000 one-second long utterances of 30 short words, spoken by thousands of people (crowdsourced on the internet).

As described at the beginning of this section, this dataset was chosen because spiking neural networks are suitable for time-varying data. However, the motivation for the use of this particular spatio-temporal benchmark extends beyond just that. Today, many mobile devices communicate with users via voice interactions, but because these devices are limited in their hardware and processing power, there are many resource constraints that these on-device speech command models must respect. For example, mobile devices have significantly less compute capabilities than most servers, so to run in real-time, on-device models must use fewer calculations than those in the cloud. Additionally, mobile devices have limited battery lives (relative to something that is always plugged in), and anything that runs continuously as do keyword spotting systems, needs to be very energy efficient or else the battery will drain exponentially quick. These constraints necessitate keyword spotting/speech command systems that involve less compute and run in very energy-efficient ways. As these are hallmark qualities of the spiking neural network, I naturally wanted to see its performance on the Google Speech Command task.
      
#### Models
Because energy and computational efficiency are not captured in the typical accuracy and loss metrics that neural network models produce, to ensure that my model bore the efficiency advantages that SNNs advertise, I replicated an SNN described in (Salaj et al., 2021) which, as shown in their study, is particularly more efficient than standard ANNs. This choice allowed for the development of a spiking neural network that could satisfy one of my goals of identifying an SNN high in biological plausibility and performance.

Another reason why I chose this neural network for this particular task was because it incorporates a biological property known as spike frequency adaptation (SFA) which allows it to perform very well on tasks that require the integration and manipulation of temporally dispersed information. Put simply, SFA denotes how the preceding firing activity of a neuron transiently increases its firing threshold and is known to occur in a substantial fraction of excitatory neurons in the neocortex (Allen Institute, 2018). In biological brains, this property is responsible for our ability to solve tasks such as recognizing a song or inverting a sequence of symbols, i.e. tasks that require the integration and manipulation of information that was dispersed across the preceding seconds, because it makes neural firing for sensory stimuli history-dependent (Pozzorini et al., 2013, 2015). In SNNs, computational SFA confers a similar type of short-term working memory, particularly when it incorporates the "negative imprinting principle" innate to biological neurons. This facilitation of short-term memory or synaptic plasticity in silico ultimately enhances the temporal computing capabilities of SNNs (Marder et al., 1996).

As might be evident at this point, an SNN with SFA functionally serves as a biological analog of the LSTM as both are capable of integrating temporally dispersed information. Naturally, since the inception of computational SFA, recurrent neural networks with SFA have been constructed, examined, and compared with LSTMs (Bellec et al., 2018). I follow this practice here.
To construct my recurrent neural networks with SFA or recurrent Long short-term Spiking Neural Networks (LSNNs), I used a Github Repository that Salaj et al. created to allow others to replicate and modulate their models. In this repository, they provide a python script that allows users to tweak, train, and evaluate an LSNN of LIF neurons on the Google Speech Commands task. This repository also provides a script to train an LSTM and a CNN making it simple to tweak and compare all three networks. The CNN model used here is the default one used in Sainath and Parada, 2015 and is included to provide additional context to the performance of the LSNN and LSTM.

#### Network Training
To train the LSNN, Salaj et al. used stochastic gradient descent in the form of Backpropagation through time (BPTT), which is a high-performing, though biologically unrealistic, optimization method. To justify its use, the authors demonstrate that almost the same performance on GSC can be achieved for SNNs through training with a more biologically realistic learning method such as e-prop (Bellec et al., 2020). Ultimately, this showed that, in terms of accuracy at least, BPTT doesn't interfere with the other biological properties of the LSNN.
 
In terms of pipeline specifications, I trained the LSNN using 2048 hidden neurons and a window stride of 1. The LSTM was trained with 512 hidden neurons, a dropout probability of 0.4, and an adam optimizer. Lastly, the CNN was trained with default parameters detailed in the methods of Sainath and Parada, 2015.


## Results

| Benchmark   | Network Type| Test Text     |
| :---        |    :----:   |          ---: |
| MNIST       | CSNN        | 98.06         |
|             | CNN         | 98.39         |
| CIFAR       | CSNN        | 70.60         |
|             | CNN         | 68.00         |
| GSC         | LSNN        | 91.20         |
|             | LSTM        | 94.40         |
|             | CNN         | 87.60         |

## Discussion
In each task, the spiking neural networks performed comparably to their artificial neural network counterparts. In the MNIST task, the SNN achieved near state-of-the-art performance on the test set which further proves the efficacy of these spiking models. Granted the use of rate-coded inputs as opposed to latency-coded inputs slightly detracted from the SNN's temporal functioning as a spiking neural network, I am confident that a latency encoded neural network would have performed only slightly worse if not the same, see Comsa et al., 2018.
 
The low performance of both my CSNN and CNN on the CIFAR-10 task was the biggest point of concern across my experiments and several theories have been devised to explain this. The most viable of such explanations is that the neural networks were not trained long enough and that the architecture of the CSNN and CNN was too simple. As the neural networks performed comparably and architecture and number of training epochs are two major features shared between the CSNN and CNN, these features likely caused the poor performance. These are also common reasons for poor testing performance in machine learning models aside from overfitting which was ruled out after plotting the accuracy over the training period. Ultimately, the CSNN and CNN could have performed better here and a comparison between the two is not fully appropriate.

On the Google Speech Commands task, the LSNN and LSTM performed exceptionally well. This was to be expected given the models used were demonstrated to perform well in Salaj et al., 2021. Nevertheless, this was still a very impressive comparison and my replication of prior results solidified the standings of LSNNs and LSTMs relative to one another. As a temporal encoded neural network, the LSNN likely exchanged performance for efficiency per the tradeoff between temporal encoding and performance commonly seen in the field (Rueckauer and Liu, 2018)
 
What is impressive is that even with this tradeoff, the LSNN is only 3.2% less accurate than the LSTM and 3.6% more accurate than the CNN. With the development of more biologically plausible training methods suited for SNNs, this gap will likely decrease and biological plausibility in the form of temporal coding will not be mutually exclusive with performance.

## Future Directions
Certain flaws or gaps in the present thesis outline future directions for this project. As my first experiment used rate-based coding to convert the MNIST dataset to spikes, in the future, I would like to use other spike encodings, particularly latency coding to fully exploit the temporal dynamics of SNNs. Another future development may include the use of datasets and hardware designed specifically for spiking neural networks, e.g. N-MNIST (Orchard et al., 2015) and neuromorphic hardware.
 
Also, since the practical use of SNNs is heavily debated, an appealing future objective of this project is to find a societally-relevant application of spiking neural networks. Currently, I am considering applying the SNN in speech recognition tasks as provided by the TIMIT and Common Voice datasets (Graves et al., 2013; Ardila et al., 2020). This task involves interpreting and translating to text various dialects and pitches of English. With these datasets and perhaps others, I hope to create an automatic speech recognition system that can comprehend dialects of English underrepresented in linguistic corpora such as AAVE (African-American Vernacular English) and English-based creoles across the global south. In this future project, I would build on my progress with the Google Speech Commands dataset and hopefully demonstrate the utility of SNNs in a widely accessible application.

## Conclusion
This thesis presents how neuroscience can serve as a source of inspiration for artificial intelligence through an analysis of spiking neural networks and their performance relative to artificial neural networks on machine learning benchmarks. In this analysis, three experiments were conducted to compare the performance of SNNs and ANNs and to determine the extent to which biological plausibility can improve neural networks. The first experiment compares the performance of a CNN and a CSNN on the MNIST task, the second compares similar networks on the CIFAR-10 task, and the last compares an LSNN, LSTM, and CNN on the Google Speech Commands Task. The performance results achieved across these experiments demonstrate that spiking neural networks perform comparably to artificial neural networks and likely carry the advantage of being more computationally efficient. That said, this thesis merely scrapes the surface of biologically plausible neural networks and further investigation is required before these networks can solidify the role of neuroscience in the development of artificial intelligence.

## List of Abbreviations
- ANN - Artificial (Analog) neural network
- BPTT -  Backpropagation-Through-Time
- CIFAR-10 - Canadian Institute for Advanced Research (10 classes)
- CNN - Convolutional neural network
- CSNN - Convolutional Spiking Neural Network
- CPU - Central processing unit
- DNN - Deep neural network
- GPU - Graphical processing unit
- GSC - Google Speech Commands
- LIF - Leaky integrate-and-fire
- LSNN - Long short-term spiking neural network
- LSTM - Long short-term memory (unit, network, etc.)
- MNIST - Modified National Institute of Standards and Technology dataset
- N-MNIST - Neuromorphic MNIST
- RAM - Random access memory
- ReLU - Rectified linear unit
- RNN - Recurrent neural network
- SFA - Spike Frequency Adaptation
- SGD - Surrogate gradient descent
- SNN - Spiking neural network
- STDP - Spike-timing-dependent plasticity
- TIMIT - Texas Instruments/Massachusetts Institute of Technology Dataset

### [References](https://docs.google.com/document/d/11ODpII-MD2wccspjFZW2LezxmHMP_yHJ6U0KpzTEFyw/edit?usp=sharing)
