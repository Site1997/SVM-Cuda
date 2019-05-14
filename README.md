# SVM-Cuda
We use simplified SMO algorithm to achieve SVM training process  

## Naive Version  
Folder **simple-smo** contains the sequential version of the code.  
Use g++ to compile and run the code.  

## CUDA Version  
Folder **cuda-svm** contains the cuda version of the code.  
Run compile_all.sh to compile the code.  
Run **cuda_svm_v0** to see speedup by **pre-compute the Kernel Matrix using CUDA** based on naive version.  
Run **cuda_svm_v1** to see speedup by **Calculate the Error Matrix using CUDA** based on v0.    
Run **cuda_svm_latest** to see speedup by **Calculate only the updates of Error Matrix using CUDA** based on v1.  

## Authors

- [Zekun Fan](https://github.com/zekunf)
- [Hanzhi Chen](https://github.com/chz15157188)
- [Site Li](https://github.com/Site1997)  
