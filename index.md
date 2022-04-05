## Abstract

(To be written once the performance metrics of the neural networks, and my interpretation thereof, are completed.) 

![Image](https://losslandscape.com/wp-content/uploads/2019/09/loss-landscape-research-9.jpg)
_Artistic visualization of the "loss landscape" of neural networks [source](https://losslandscape.com/)_

## Introduction

The human brain serves as the ultimate inspiration for the field of artificial intelligence. Of the many "intelligent" machine learning algorithms and computational architectures that have adopted properties of our nervous system to achieve the brain’s efficiency, artificial neural networks (ANNs) have most explicitly done so, successfully mimicking many representations of brain function. Put simply, an artificial neural network comprises a means of performing machine learning and thus learns from data to predict and classify information. The power of this technique, however, often exceeds that of other machine learning algorithms as evidenced in its preferential use by state of the art artificial intelligence systems, such as Apple's speech recognition ([Aron, 2011](https://doi.org/10.1016/S0262-4079(11)62647-X)) and DeepMind's AlphaFold ([Jumper et al., 2021](https://doi.org/10.1038/s41586-021-03819-2)). This has led many to believe that the brain-inspired approach of neural networks has conferred onto the algorithm a human-like ability to recognize patterns and that biological plausibility may be a guide to further advances in artificial intelligence.

Given the brain embodies many of the functions we wish to see in deep learning technology, it is reasonable to believe that the principles of neuroscience can serve as a guide in the development of more robust deep learning and intelligent systems. In spite of this, many of today's artificial neural networks are still vague representations of the biological neural networks they were originally modeled after and the differences between the two are increasingly outweighing the similarities. This growing deviation of artificial neural networks from biological neural networks is further exacerbated by a lack of collaboration between the fields of neuroscience and artificial intelligence which is requisite for the development of biologically powerful algorithms. To confront these challenges, the present project investigates and demonstrates how neuroscience can serve as a source of inspiration for artificial intelligence via biologically-plausible Neural Networks and the applications thereof.

Within this investigation of ways in which biological plausibility may improve neural networks, an existing algorithm known as a spiking neural network (SNN) will be discussed in relatively greater depth and comprise the majority of my engagement with the various routes to biological plausibility in deep learning algorithms. SNNs are artificial neural networks that more closely mimic biological neural networks in that their neurons send information on their own schedule relative to others (i.e. asynchronously), particularly when their computational equivalent of a membrane reaches a specific value known as the threshold. Given the progress made by SNNs in achieving a biologically plausible architecture, they serve as a great foundation for a discussion on what biological plausibility in neural networks may look like and what direction should be trekked to improve upon what currently works. 

Before proceeding, it’s worth noting that there exists three main limitations in the present project thesis. First, the pursuit of biological plausibility will be inherently limited by the constraints of in-silico development, so while the standards of our biological-plausible networks are high, they reasonably fit within these constraints. The second limitation constitutes the fact that most comprehensive software packages for simulating SNNs were developed to model neural circuits and not target machine learning applications as is intended in this project. In spite of this, there exist a handful of solid, although uncommon, software packages that can be used for machine learning applications. Lastly, it has been shown that neural networks with a great degree of biological realism are not appropriate for applications in machine learning since they are computationally expensive to simulate. While this represents a major obstacle to the present goal, it is unlikely that the neural networks to be used will reach the magnitude that will push the limits of the standard GPU (Hazan et al., 2018).

## Spiking Neural Networks

(**Working on this transition**)

Convolutional neural networks and recurrent neural networks demonstrate that neuroscience can serve as a source of inspiration for artificial neural network architecture, furthermore, that biologically inspired algorithms can exceed the performance of conventional ones.

However, when it comes to the pursuit of biological-plausibility, it’s worth noting that classic CNNs and RNNs are fairly still artificial and can only claim bio-inspiration— a word for which there exists very little criteria. Because the standard for what constitutes bio-inspiration is low, many of today's “bio-inspired” neural networks merely pick and choose which parts of neurons to mimic (with varying extents) and thus do not comprehensively demonstrate whether neural computation in and of itself can improve artificial neural networks. Investigating this question requires stepping beyond bio-inspiration in neural networks and into biological plausibility, that is, neural networks that do not selectively replicate parts of biology, but rather aim to encompass all properties of it. Granted, building intelligent machines in silico inherently limits the extent to which artificial neurons can mimic biological ones, replication of the algorithmic levels of neural computation is feasible and represents a viable goal in the pursuit of biological plausibility. As the field stands today, spiking neural networks serve as a logical first step to this goal.

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

My project goal is to identify spiking network architectures where performance and biological plausibility are directly correlated and, particularly cases in which biological plausibility is the cause of exceptional performance. This will entail experimentally comparing the average performance of SNNs with their conventional DNN counterparts on several benchmarks to assess whether there is an advantage in the mere use of temporally dynamic spikes as opposed to conventional DNNs. 

These benchmarks include:
1. N-MNIST
2. CIFAR-10 - DVS
3. NORB
5. DVS-Gesture

## Results

- [In progress](https://github.com/chiralevy/Deep-Learning-with-Biologically-Plausible-Neural-Networks/blob/main/MNIST_SNN.ipynb)

## Discussion

## Ethical Concerns

## Reflection - Epilogue
