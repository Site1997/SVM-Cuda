#ifndef CUDA_SVM_H
#define CUDA_SVM_H

#include <cstdio>
#include <cstdlib>
#include <cassert>

#define malloc2D(name, xDim, yDim, type) do {               \
    name = (type **)malloc(xDim * sizeof(type *));          \
    assert(name != NULL);                                   \
    name[0] = (type *)malloc(xDim * yDim * sizeof(type));   \
    assert(name[0] != NULL);                                \
    for (size_t i = 1; i < xDim; i++)                       \
        name[i] = name[i-1] + yDim;                         \
} while (0)

void cuda_svm(int objs,int coords,double** x,int* y,double c,int max_passes,double* a,double* b_out);
void cuda_svm_predict(int objs,int coords,double** x,int objs_train,double** x_train, double* a,double b, int* y_train, int* y_out);

#endif