#include "cuda_svm.h"
// #include <cuda_runtime.h>

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

//calE is done within training, using kvals
double calE(int objs,double* a,double b,int* y,double** kval,int target){
    double fx = b;
	for (int i=0; i<objs; i++) {
		fx += a[i] * y[i] * kval[i][target];
	}
	return fx - y[target];
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

double calIta(double *xi, double *xj,int xDim) {
	double ita = 0;
	for (int d=0; d<xDim; d++) {
		ita += 2*xi[d]*xj[d] - xi[d]*xi[d] - xj[d]*xj[d];
	}
	return ita;
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

// loss = regularization(a) + C * loss(a; x,y)
// C ranges from 1e-5 ~ 1e5
void cuda_svm(int objs,int coords,double** x,int* y,int c,int max_passes,double* a,double* b_out){
    int b=0;

    int pass=0;

    double** kval;
    malloc2D(kval,objs,objs,double);

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
        double* kval_d;
        cudaMalloc(&kval_d, objs*objs*sizeof(double));

        calc_linear_kernel<<<objs*objs/256+1,256>>>(objs,coords,x_r_d,kval_d);
        cudaMemcpy(kval[0],kval_d,objs*objs*sizeof(double),cudaMemcpyDeviceToHost);
    }

    // for (int i=0;i<objs;++i)
    //   for (int j=0;j<objs;++j)
    //     printf("%.2f ",kval[i][j]);

    while (pass < max_passes) {
		int num_changed_alphas = 0;
		for (int i=0; i<objs; i++) {
			double ei = calE(objs,a,b,y,kval,i);
			if ((y[i]*ei < -EPS && a[i] < c) || (y[i]*ei > EPS && a[i] > 0)) {
                //updated rand method
                int j = rand() % (objs-1);
                j=(j>=i)?j+1:j;

				double ej = calE(objs,a,b,y,kval,j);
				double ai_old = a[i], aj_old = a[j];
				double lb = 0, rb = c;
				calLH(c,y[i], y[j], a[i], a[j], lb, rb);
				if (abs(lb - rb) < EPS)
					continue ;
				double ita = calIta(x[i], x[j],coords);
				if (ita >= 0)
					continue ;
				a[j] = computeAjAndClipValue(a,y, ei, ej, ita, lb, rb, j);
				if (abs(a[j] - aj_old) < 1e-5)
					continue ;
                a[i] = determineAi(a,y, i,j, aj_old);
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
			}
		}
		if (num_changed_alphas == 0) pass ++;
		else pass = 0;
    }
    *b_out=b;
}