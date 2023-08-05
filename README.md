# DMP_simulation
A simulation project on Dynamic Movement Primitive (DMP) , containing 1D numerical simulation, 2D learning from demonstration and 3D simulation for robotic arm.

<font color="red">标签中的字会显示为红色</font> 之后的字不会再显示为红色

## 1. About DMP
Supported by several experimental findings, that biological systems are able to combine and adapt basic units of motion into complex tasks, which finally lead to the formulation of the motor primitives theory. In this respect, Dynamic Movement Primitives (DMPs) represent an elegant mathematical formulation of the motor primitives as stable dynamical systems, and are well suited to generate motor commands for artificial systems like robots.

DMP was initially proposed by Schaal et al. in 2002, as the most classical theories for implementing robot imitation learning. Its various extended forms and wide-ranging applications have been prevalent in research over the past two decades.
DMP is based on a set of nonlinear differential equations to describe motion trajectories and utilizes a group of Radial Basis Functions (RBFs) for approximation and generation of movements. The DMP model can be applied to various motions, including robotic arm movements, robot path planning, human motion imitation, handwriting and speech generation, among others. 

Classic DMPs include **discrete** and **rhythmic** types, primarily composed of two components: **the forcing term** and **the feedback term**:
- The forcing term describes the desired movement trajectory, which is typically obtained through demonstrations or manually set target trajectories. It is used to guide the generated movement along the desired trajectory.
- The feedback term is employed for adaptive control, enabling the generated movement to adapt to changes and disturbances in the environment. The introduction of the feedback term provides the DMP model with robustness and adaptability.

The generation process of the DMP model includes two stages: training and reproduction. In the training stage, the parameters of the basis functions and the weights of the feedback term are adjusted by learning from the demonstrated trajectories, resulting in a suitable DMP model. In the reproduction stage, the initial conditions, motion goals, spatial scaling, and temporal scaling can be flexibly set according to the requirements. Finally, by computing the output of the DMP model using the above parameters, it is possible to achieve the reproduction or generalization of the demonstrated trajectories.

For specific theoretical details and the current researchs of DMP, you can refer to the survey: 
> Saveriano, Matteo, et al. "Dynamic movement primitives in robotics: A tutorial survey." arXiv preprint arXiv:2102.03861 (2021).

## 2. About this project
After the theoretical learning, I built the following three demos to further deepen the understanding and application of DMP:
- Verify the DMP model through 1D-trajectory numerical simulation
- Understanding how DMP guides a simple system to perform 2D-trajectory demonstration learning;
- Obtaining data of virtual robotic arm through 3D simulation, and exploring its implementation of demonstration learning based on DMP.

## 3. Development environment
Windows10(no GPU) + PyCharm(2023.2) + python3.9

## 4. Simulations
### 1D numerical simulation
