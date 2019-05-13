#include "cuda_svm.h"
#include <iostream>
using std::cout;

int sampleNum, xDim;
double** x;
int* y;
double* a;
double b=-1;

double** x_test;
int* y_test;
int objs_test;

void input(char* fpath,double*** p_x,int** p_y,int* objs,int* coords) {
	FILE* f=fopen(fpath, "r");
	int sampleNum, xDim;
	double** x;
	int* y;
	fscanf(f,"%d%d", &sampleNum, &xDim);
    malloc2D(x,sampleNum,xDim,double);
    y=(int*)calloc(sampleNum,sizeof(double));
	for (int i=0; i<sampleNum; i++) {
		for (int j=0; j<xDim; j++)
			fscanf(f,"%lf", &x[i][j]);
		fscanf(f,"%d", &y[i]);
		// printf("%.2f %.2f %d\n", x[i][0], x[i][1], y[i]);
	}
	*p_x=x;
	*p_y=y;
	*objs=sampleNum;
	*coords=xDim;
	fclose(f);
}

void eval() {
	int *pred=(int*)calloc(objs_test,sizeof(int));
	cuda_svm_predict(objs_test,xDim,x_test,sampleNum,x,a,b,y,pred);

	int correctNum = 0;
	int posNum = 0;
	for (int i=0; i<objs_test; i++) {
		int curPred = pred[i];
		correctNum += (curPred == y_test[i]);
		posNum += curPred >=0? 1 : 0;
	}
	puts("Testing Results:");
	printf("Positive number of Samples: %d\n", posNum);
	printf("Negative number of Samples: %d\n", objs_test - posNum);
	printf("Final Accuracy : %.4f\n", (double)correctNum / (double)(objs_test));
}

int main(int argc,char *argv[]){
	if (argc==4){
		input(argv[1],&x,&y,&sampleNum,&xDim);
		double TRAIN_RATIO=0.8;
		sscanf(argv[3],"%lf",&TRAIN_RATIO);

		//Train_test_split
		objs_test=sampleNum-(sampleNum*TRAIN_RATIO+1);
		sampleNum=sampleNum-objs_test;
		x_test=x+sampleNum;
		y_test=y+sampleNum;
		a=(double*)calloc(sampleNum,sizeof(double));
		cuda_svm(sampleNum,xDim,x,y,0.1,1000,a,&b);
		eval();
	}else if (argc==3) {
		input(argv[1],&x,&y,&sampleNum,&xDim);
		input(argv[2],&x_test,&y_test,&objs_test,&xDim);
		a=(double*)calloc(sampleNum,sizeof(double));
		printf("%d %d",sampleNum,xDim);
		cuda_svm(sampleNum,xDim,x,y,0.1,1000,a,&b);
		eval();
	}else{
		puts("cuda_svm usage:\n ./cuda_svm <train_file> <test_file>\n ./cuda_svm <input_file> split <train_ratio>\n");
		puts("Examples:\n ./cuda_svm ../data/a9a_flat ../data/a9a_flat.t\n ./cuda_svm ../data/a9a_flat split 0.8\n");
	}
    return 0;
}