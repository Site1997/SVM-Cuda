#include "cuda_svm.h"

__global__ static
void calc_linear_kernel_predict(int objs,int coords,double* x,int objs_train,double* x_train,double* out){
    int id=blockDim.x * blockIdx.x + threadIdx.x;
    int i=id/objs;
    int j=id%objs;
    if (i<objs_train){
        double r=1.0;
        for (int k=0;k<coords;k++){
            r += x_train[coords*i+k] * x[coords*j+k];
        }
        out[id]=r;
    }
}

__global__ static
void calc_predict(int objs,int objs_train,double* a,double b,int* y_train,double* kval,int* y){
    int id=blockDim.x * blockIdx.x + threadIdx.x;
    if (id<objs){
        double fx=b;
        for (int i=0;i<objs_train;i++){
            //access to a and y are not coalesced
            fx+=a[i]*y_train[i]*kval[i*objs+id];
        }
        y[id] = fx>=0 ? 1:-1;
    }
}

void cuda_svm_predict(int objs,int coords,double** x,int objs_train,double** x_train, double* a,double b, int* y_train, int* y_out){
    //kval_d[objs_train][objs]
    double* kval_d;
    cudaMalloc(&kval_d, objs*objs_train*sizeof(double));

    {
        double* x_d;
        cudaMalloc(&x_d,objs*coords*sizeof(double));
        cudaMemcpy(x_d,x[0],coords*objs*sizeof(double),cudaMemcpyHostToDevice);
        
        double* x_train_d;
        cudaMalloc(&x_train_d, coords*objs_train*sizeof(double));
        cudaMemcpy(x_train_d,x_train[0],coords*objs_train*sizeof(double),cudaMemcpyHostToDevice);

        calc_linear_kernel_predict<<<objs*objs_train/256+1,256>>>(objs,coords,x_d,objs_train,x_train_d,kval_d);

        cudaFree(x_d);
        cudaFree(x_train_d);
    }

    // double* kval=(double*)calloc(objs*objs_train,sizeof(double));
    // cudaMemcpy(kval,kval_d,objs*objs_train*sizeof(double),cudaMemcpyDeviceToHost);
    // for (int i=0;i<objs*objs_train;++i){
    //     printf("%f ",kval[i]);
    // }

    double* a_d;
    cudaMalloc(&a_d,objs_train*sizeof(double));
    cudaMemcpy(a_d,a,objs_train*sizeof(double),cudaMemcpyHostToDevice);

    int* y_train_d;
    cudaMalloc(&y_train_d,objs_train*sizeof(int));
    cudaMemcpy(y_train_d,y_train,objs_train*sizeof(int),cudaMemcpyHostToDevice);

    int* y_d;
    cudaMalloc(&y_d,objs*sizeof(int));

    calc_predict<<<objs/256+1,256>>>(objs,objs_train,a_d,b,y_train_d,kval_d,y_d);
    cudaMemcpy(y_out,y_d,objs*sizeof(int),cudaMemcpyDeviceToHost);
    // for (int i=0;i<objs;++i){
    //     printf("%d ",y_out[i]);
    // }

    cudaFree(kval_d);
    cudaFree(a_d);
    cudaFree(y_train_d);
    cudaFree(y_d);
}