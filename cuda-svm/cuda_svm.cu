#include "cuda_svm.h"
// #include <cuda_runtime.h>
#include <time.h>

const double EPS = 0.0001;

//possible optimization: load x[][i] into shared and parallel on j
// x[coords][objs], y[objs], out[objs][objs]
__global__ static
void calc_linear_kernel(int objs,int coords,double* x,double* out){
    int id=blockDim.x * blockIdx.x + threadIdx.x;
    int i=id/objs;
    int j=id%objs;
    if (i<objs){
        
        double r=0.0;
        for (int k=0;k<coords;k++){
            r+=x[objs*k+i]*x[objs*k+j];
        }
        out[objs*i+j]=r;
    }
}

__global__ static
void calc_e(int objs,double* a,double b,int* y,double* kval,double* e){
    int id=blockDim.x * blockIdx.x + threadIdx.x;
    if (id<objs){
        double fx=b;
        for (int i=0;i<objs;i++){
            //access to a and y are not coalesced
            fx+=a[i]*y[i]*kval[i*objs+id];
        }
        e[id]=fx-y[id];
    }
}

// solid 13 arguments......
__global__ static
void update_e(int objs,double* e,double* kval,double b_old,double b_new,int i,int j,int yi,int yj,double ai_old,double ai_new,double aj_old,double aj_new){
    int id=blockDim.x * blockIdx.x + threadIdx.x;
    if (id<objs){
        double val=e[id];
        val+=(b_new-b_old);
        double ti=yi*kval[i*objs+id];
        double tj=yj*kval[j*objs+id];
        val += ti*(ai_new-ai_old);
        val += tj*(aj_new-aj_old);
        e[id]=val;
    }
}

void calLH(double C,int yi, int yj, double ai, double aj, double &lb, double &rb)  {
	if (yi != yj) {
		lb = max(0., aj - ai);
		rb = min(C, C + aj - ai);
	}
	else {
		lb = max(0., ai + aj - C);
		rb = min(C, ai + aj);
	}
}

double computeAjAndClipValue(double* a, int* y,double ei, double ej, double ita, double lb, double rb, int j) {
	double aj = a[j] - y[j] * (ei - ej) / ita;
	if (aj > rb) aj = rb;
	else if (aj < lb) aj = lb;
	return aj;
}

double determineAi(double* a,int* y,int i, int j, double aj_old) {
	return a[i] + y[i] * y[j] * (aj_old - a[j]);
}

void compute_eVal(int objs,double b, double* a_d,int* y_d,double* kval_d,double* eVal_d,double* eVal){
    //Might be costly to copy all of a since only a tiny part of a is updated
    calc_e<<<objs/256+1,256>>>(objs,a_d,b,y_d,kval_d,eVal_d);
    cudaMemcpy(eVal,eVal_d,sizeof(double)*objs,cudaMemcpyDeviceToHost);
}

inline void update_eVal(int objs,double* eVal_d,double* kval_d,double b_old,double b_new,int i,int j,int yi,int yj,double ai_old,double ai_new,double aj_old,double aj_new,double* eVal){
    //Might be costly to copy all of a since only a tiny part of a is updated
    update_e<<<objs/256+1,256>>>(objs,eVal_d,kval_d,b_old,b_new,i,j,yi,yj,ai_old,ai_new,aj_old,aj_new);
    cudaMemcpy(eVal,eVal_d,sizeof(double)*objs,cudaMemcpyDeviceToHost);
}

// loss = regularization(a) + C * loss(a; x,y)
// C ranges from 1e-5 ~ 1e5
void cuda_svm(int objs,int coords,double** x,int* y,double c,int max_passes,double* a,double* b_out){
    int b=0;

    int pass=0;

    double** kval;
    malloc2D(kval,objs,objs,double);
    double* kval_d;
    cudaMalloc(&kval_d, objs*objs*sizeof(double));
    double* a_d;
    cudaMalloc(&a_d,objs*sizeof(double));
    cudaMemcpy(a_d,a,objs*sizeof(double),cudaMemcpyHostToDevice);

    //May possibly use Constant memory, if switched to byte*
    int* y_d;
    cudaMalloc(&y_d,objs*sizeof(int));
    cudaMemcpy(y_d,y,objs*sizeof(int),cudaMemcpyHostToDevice);

    //Pre calculate kernel via cuda
    {
        double** x_r;
        malloc2D(x_r,coords,objs,double);
        for (int i=0;i<coords;i++)
            for (int j=0;j<objs;j++)
                x_r[i][j]=x[j][i];
        
        double* x_r_d;
        cudaMalloc(&x_r_d, coords*objs*sizeof(double));
        cudaMemcpy(x_r_d,x_r[0],coords*objs*sizeof(double),cudaMemcpyHostToDevice);

        calc_linear_kernel<<<objs*objs/256+1,256>>>(objs,coords,x_r_d,kval_d);
        cudaMemcpy(kval[0],kval_d,objs*objs*sizeof(double),cudaMemcpyDeviceToHost);

        free(x_r[0]);
        free(x_r);
        cudaFree(x_r_d);
    }


    // FILE* fk=fopen("custom.txt","w");
    // for (int i=0;i<objs;i++){
    //     double ei = calE(objs,a,b,y,kval,i);
    //     fprintf(fk,"e[%d]=%f\n",i,ei);
    // }
    // fclose(fk);

    // for (int i=0;i<objs;++i)
    //   for (int j=0;j<objs;++j)
    //     printf("%.2f ",kval[i][j]);

    double* eVal=(double*)calloc(objs,sizeof(double));
    double* eVal_d;
    cudaMalloc(&eVal_d,objs*sizeof(double));

    compute_eVal(objs,b,a_d,y_d,kval_d,eVal_d,eVal);


    int iter=0;
    const int max_iter=50;
    while (pass < max_passes && iter < max_iter) {
        double st_clk=(double)clock()/CLOCKS_PER_SEC;
		int num_changed_alphas = 0;
		for (int i=0; i<objs; i++) {
            double ei = eVal[i];
            // printf("e[%d]=%f\n",i,ei);
			if ((y[i]*ei < -EPS && a[i] < c) || (y[i]*ei > EPS && a[i] > 0)) {
                //updated rand method
                int j = rand() % (objs-1);
                j=(j>=i)?j+1:j;

				double ej = eVal[j];
				double ai_old = a[i], aj_old = a[j];
				double lb = 0, rb = c;
				calLH(c,y[i], y[j], a[i], a[j], lb, rb);
				if (abs(lb - rb) < EPS)
					continue ;
				double ita = 2*kval[i][j] - kval[i][i] -kval[j][j];
				if (ita >= 0)
					continue ;
				a[j] = computeAjAndClipValue(a,y, ei, ej, ita, lb, rb, j);
				if (abs(a[j] - aj_old) < 1e-5)
					continue ;
                a[i] = determineAi(a,y, i,j, aj_old);
                double b_old=b;
                //updateB inlined here for convenience
                {
                    double b1 = b - ei - y[i]*(a[i]-ai_old)*kval[i][i] - y[j]*(a[j]-aj_old)*kval[i][j];
                    double b2 = b - ej - y[i]*(a[i]-ai_old)*kval[i][j] - y[j]*(a[j]-aj_old)*kval[j][j];
                    double finalB = 0;
                    if (0 < a[i] && a[i] < c) finalB = b1;
                    else if (0 < a[j] && a[j] < c) finalB = b2;
                    else finalB = (b1 + b2) / 2;
                    b=finalB;
                }
                num_changed_alphas ++;
                update_eVal(objs,eVal_d,kval_d,b_old,b,i,j,y[i],y[j],ai_old,a[i],aj_old,a[j],eVal);
			}
        }
        // printf("changed: %d\n",num_changed_alphas);
		if (num_changed_alphas == 0) pass ++;
        else pass = 0;
        double ed_clk=(double)clock()/CLOCKS_PER_SEC;
        printf("iter %d,changed %d,runtime: %f\n",iter,num_changed_alphas,ed_clk-st_clk);
        ++iter;
    }
    *b_out=b;
    free(kval[0]);
    free(kval);
    cudaFree(kval_d);
    cudaFree(a_d);
    cudaFree(y_d);
    
    free(eVal);
    cudaFree(eVal_d);
    
}

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