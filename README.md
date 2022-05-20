## Paths to a Less Artificial Intelligence

## Introduction

The human brain serves as the ultimate inspiration for the field of artificial intelligence. Of the many machine learning algorithms that have adopted properties of our nervous system to achieve its efficiency, the most renowned is the artificial neural network (ANN). Inspired by the depth and connectivity of biological neural networks, ANNs use layers of computational neurons to perform a distinct form of machine learning called "deep learning". This technique allows them to learn from colossal amounts of data to predict and classify information with great accuracy. The power of this technique often exceeds that of other machine learning algorithms as evidenced in a wealth of literature and its preferential use by state-of-the-art artificial intelligence systems such as Apple's Siri ([Aron, 2011](https://doi.org/10.1016/S0262-4079(11)62647-X)) and DeepMind's AlphaFold ([Jumper et al., 2021](https://doi.org/10.1038/s41586-021-03819-2)). This distinction has led me to believe that the brain-inspired approach of neural networks has conferred onto the algorithm a human-like ability to recognize patterns and that biological plausibility may be a guide to further advances in artificial intelligence.

Given the brain embodies many of the functions we wish to see in deep learning technology, it is reasonable to believe that the principles of neuroscience can serve as a guide in the development of more robust deep learning and intelligent systems. Despite this, many of today's artificial neural networks are still vague representations of the biological neural networks they were originally modeled after decades ago and the differences between the two are increasingly outweighing the similarities. This growing deviation of artificial neural networks from biological neural networks is further exacerbated by a lack of communication between the fields of neuroscience and computer science, which is necessary for the development of biologically powerful neural networks. To confront these challenges, the present thesis investigates how neuroscience can serve as a source of inspiration for artificial intelligence through biologically-plausible neural networks and the applications thereof.

Within this investigation of ways in which biological plausibility may improve neural networks, an existing algorithm known as a spiking neural network (SNN) will be discussed in relatively greater depth and comprise the majority of my engagement with the various routes to biological plausibility in deep learning algorithms. SNNs are artificial neural networks that more closely mimic biological neural networks in that their neurons send information on their own schedule relative to others (i.e. asynchronously), particularly when their computational equivalent of a membrane reaches a specific value known as the threshold. Given the progress made by SNNs in achieving a biologically plausible architecture, they serve as a great foundation for a discussion on what biological plausibility in neural networks may look like and what direction should be trekked to improve upon what currently works. 

Before proceeding, one limitation of this project thesis worth noting is that neural networks with great degrees of biological realism are not appropriate for applications in machine learning since they are computationally expensive to simulate in traditional von-Neumann CPUs. For this reason, the pursuit of biological plausibility will be limited by the constraints of in-silico development. While this represents a major obstacle to the present goal, it is unlikely that the neural networks to be used will reach a complexity that will push the limits of the standard GPU or CPU (Hazan et al., 2018).

## Spiking Neural Networks

As described above, spiking neural networks largely differ from today's conventional deep neural networks in that they incorporate the temporal dimension of neural computation. That is, instead of forwardly propagating information synchronously as is typically done by default, the neurons of an SSN transmit info only when their computational equivalent of a membrane potential— the membrane's electrical charge— reaches a threshold. Once this threshold is achieved, this neuron "spikes" (typically represented as a zero or one in an activation function) and sends a signal to other neurons which in turn influences their membrane potential. Depending on the implementation of the SSN, after this, the value of the neuron can briefly drop below its initial activation to mimic a biological neuron’s refractory period after an action potential. In operation, multiple spikes at discrete points in time create the appearance of “spike trains” which are a series of spikes that are inputted into a network and produced as output. At the end, whether the neuron fired or did not receive the sufficient number of spikes to fire, the value of the neuron will gradually return to its baseline ([Ponulak & Kasiński, 2010](https://pubmed.ncbi.nlm.nih.gov/22237491/)). 

Over the years, various models have been developed to model the SSN. The most popular and the one to be used here is the Leaky Integrate and Fire threshold model which simply proposes setting the value in a neuron to its activation level and then having incoming spike trains increase or decrease this value until the threshold is reached or the state decays. The main goal in this model is to train the synapses (the weights) to cause the postsynaptic neuron to generate a spike train with desired spike times.

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

My project goal is to construct SNNs whose performance and biological plausibility are not strongly inversely correlated and perform as well as, if not better than, ANNs on benchmark tasks. In addition, I hope to address the limitations of the aforementioned SNNs to further investigate the extent to which biological plausibility can benefit neural network architectures. This will entail comparing the performance of SNNs high in biological plausibility with their traditional ANN counterparts on benchmark tasks.

These benchmarks include:
1. MNIST (Lecun et al., 1998)
2. CIFAR-10 (Krizhevsky, 2009)
3. Google Speech Commands (Warden, 2018)

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
