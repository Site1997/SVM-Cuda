#include "cuda_svm.h"
#include <iostream>
using std::cout;

int sampleNum, xDim;
double** x;
int* y;
double* a;
double b=-1;
const double TRAIN_RATIO = 0.8;

void input() {
	freopen("../data/a9a_flat", "r", stdin);
	scanf("%d%d", &sampleNum, &xDim);
    malloc2D(x,sampleNum,xDim,double);
    y=(int*)calloc(sampleNum,sizeof(double));
    a=(double*)calloc(sampleNum,sizeof(double));
	for (int i=0; i<sampleNum; i++) {
		for (int j=0; j<xDim; j++)
			scanf("%lf", &x[i][j]);
		scanf("%d", &y[i]);
		// printf("%.2f %.2f %d\n", x[i][0], x[i][1], y[i]);
	}
}

double calFx(int m, double *xTarget) {
	double fx = 0;
	for (int i=0; i<m; i++) {
		double ai_yi = a[i] * y[i];
		double innerProd = 0;
		for (int d=0; d<xDim; d++) {
			innerProd += x[i][d] * xTarget[d];
		}
		fx += ai_yi * innerProd;
	}
	fx += b;
	return fx;
}

void eval() {
	int correctNum = 0;
	int m = sampleNum * TRAIN_RATIO + 1;
	int posNum = 0;
	for (int i=m; i<sampleNum; i++) {
		double fx = calFx(m, x[i]);
		int curPred = fx >= 0? 1 : -1;
		correctNum += (curPred == y[i]);
		posNum += fx >=0? 1 : 0;
	}
	puts("Testing Results:");
	printf("Positive number of Samples: %d\n", posNum);
	printf("Negative number of Samples: %d\n", (sampleNum-m) - posNum);
	printf("Final Accuracy : %.4f\n", (double)correctNum / (double)(sampleNum-m));
}

int main(){
    input();
    cuda_svm(sampleNum*TRAIN_RATIO+1,xDim,x,y,0.1,1000,a,&b);
    // cout<<b;
    // for (int i=0;i<sampleNum;++i)
    //   cout<<a[i]<<' ';
	eval();
    return 0;
}