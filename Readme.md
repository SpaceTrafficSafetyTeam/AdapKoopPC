#  HLTP: A Cognitive-Based Trajectory Prediction Approach for Autonomous Driving

## Overview
The code repository for the study titled **"A scalable adaptive deep Koopman predictive controller for real-time optimization of mixed traffic flow"** will be made available upon acceptance of the manuscript.



## Main Contributions

- An adaptive deep Koopman network (AdapKoopnet) is proposed for modeling HDVs car-following behavior with the high-dimensional linear model; 
- A scalable state prediction model of mixed traffic flow is developed by integrating linear dynamic model of CAVs and linear prediction blocks from AdapKoopnet;
- An adaptive deep Koopman predictive control framework (AdapKoopPC) based on the state prediction model is proposed for mitigating traffic oscillations in mixed traffic flow.



## Abstract

The use of connected automated vehicle (CAV) is advocated to mitigate traffic oscillations in mixed traffic flow consisting of CAVs and human driven vehicles (HDVs). This study proposes an adaptive deep Koopman predictive control framework (AdapKoopPC) for regulating mixed traffic flow. Firstly, a Koopman theory-based adaptive trajectory prediction deep network (AdapKoopnet) is designed for modeling HDVs car-following behavior. AdapKoopnet enables the representation of HDVs behavior by a linear model in a high-dimensional space. Secondly, the model predictive control is employed to smooth the mixed traffic flow, where the combination of the linear dynamic model of CAVs and linear prediction blocks from AdapKoopnet is embedded as the predictive model into the AdapKoopPC. Finally, the predictive performance of the prosed AdapKoopnet is verified using the HighD naturalistic driving dataset. Furthermore, the control performance of AdapKoopPC is validated by the numerical simulations. Results demonstrate that the AdapKoopnet provides more accuracy HDVs predicted trajectories than the baseline nonlinear models. Moreover, the proposed AdapKoopPC exhibits more effective control performance with less computation cost compared with baselines in mitigating traffic oscillations, especially at the low CAVs penetration rates. The code of proposed AdapKoopPC is open source.



## Framework

**“The model architecture of AdapKoopnet”** .
![framework](https://github.com/SpaceTrafficSafetyTeam/PeMTFLN/blob/main/framework.pdf)




## Environment

- **Operating System**: Ubuntu 20.04
- **CUDA Version**: 11.4



## Train



## Evaluation



 


