# CNC_toy
Toy SNN net to test minimum necessary for anomaly detection.
The net is a 12 neurons network with 10 leaky integrate-and-fire and 2 output neurons.
The hidden layer has different beta values from 0.4 to 0.8 to act as a bank of low pass filters.
This choice allows the network to classify different vibration armonics to detect failures.
This repo depends on snntorch and the CNC machining dataset from Bosch:
Paper https://doi.org/10.1016/j.procir.2022.04.022
Repo for the dataset https://github.com/boschresearch/CNC_Machining
Once downloaded, put the dataset folder in the same folder of CNC_toy.


