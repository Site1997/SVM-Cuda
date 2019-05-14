# SVM-Cuda
We use simplified SMO algorithm to achieve SVM training process  

## Naive Version  
Folder **simple-smo** contains the sequential verion of the code.  
Use g++ to compile and run the code.  

## CUDA Version  
Folder **cuda-svm** contains the cuda verion the code.  
Run compile_all.sh to compile the code.  
Run **cuda_svm_v0** to exploit speedup by **pre-compute the Kernel Matrix using CUDA** based on naive version.  
Run **cuda_svm_v1** to exploit speedup by **Calculate the Error Matrix using CUDA** based on v0.    
Run **cuda_svm_latest** to exploit speedup by **Calculate only the updates of Error Matrix using CUDA** based on v1.  
