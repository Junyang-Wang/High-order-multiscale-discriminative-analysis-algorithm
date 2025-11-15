# High-order-multiscale-discriminative-analysis-algorithm
The code is the Matlab version for fast, powerful, and universally consistent independence testing.
The Matlab version of this code is Matlab R2020b with the required toolboxes, tensor_toolbox-v3.4, eeglab2022.1, and covariance toolbox-master.

Author: Xiaoyu Zhou, xiaoyu_zhou@tju.edu.cn


1. Demo for use
a. Instructions to run on data: Open MATLAB with the required toolboxes installed, navigate to the offline_decoding folder, and run the main function.

b. Expected output: 
Mdl_HMDA: offline model based on HMDA algorithm
ACC: the offline cross-validation classification accuracy for the 9 basic directional commands.

c. Expected run time for demo on a "normal" desktop computer: about 5 mins.


2. Instructions for use
a. Reproduction instructions.
The output ACC represents the average over five repetitions of five-fold cross-validation, providing high stability and reproducibility.
