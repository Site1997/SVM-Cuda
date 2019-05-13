#include <queue>
#include <cmath>
#include <cstdio>
#include <vector>
#include <cstring>
#include <iostream>
#include <algorithm>
using namespace std;

const int MAXN = 32768;
const int MAXCOORD = 128;
const int MAX_PASSES = 1000;
const double TRAIN_RATIO = 0.5;
const double EPS = 0.0001;
// loss = regularization(a) + 𝐶 * loss(a; x,y)
// C ranges from 1e-5 ~ 1e5
const double C = 0.1;

// Below is the baseline accuracy from the Naive Implementation:
// TRAIN_RATIO 0.2 Accuracy: 65.82%
// TRAIN_RATIO 0.9 Accuracy: 100%

int sampleNum, xDim;
double x[MAXN][MAXCOORD]; int y[MAXN];
double a[MAXN], b;

void init() {
	sampleNum = xDim = 0;
	b = 0;
	memset(a, 0, sizeof(a));
	memset(x, 0, sizeof(x));
	memset(y, 0, sizeof(y));
}

void input() {
	freopen("../data/a9a_flat", "r", stdin);
	scanf("%d%d", &sampleNum, &xDim);
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

double calE(int m, double *xTarget, double yTarget) {
	return calFx(m, xTarget) - yTarget;
}

void calLH(int yi, int yj, double ai, double aj, double &lb, double &rb)  {
	if (yi != yj) {
		lb = max(0., aj - ai);
		rb = min(C, C + aj - ai);
	}
	else {
		lb = max(0., ai + aj - C);
		rb = min(C, ai + aj);
	}
}

double calIta(double *xi, double *xj) {
	double ita = 0;
	for (int d=0; d<xDim; d++) {
		ita += 2*xi[d]*xj[d] - xi[d]*xi[d] - xj[d]*xj[d];
	}
	return ita;
}

double computeAjAndClipValue(double ei, double ej, double ita, double lb, double rb, int j) {
	double aj = a[j] - y[j] * (ei - ej) / ita;
	if (aj > rb) aj = rb;
	else if (aj < lb) aj = lb;
	return aj;
}

double determineAi(int i, int j, double aj_old) {
	return a[i] + y[i] * y[j] * (aj_old - a[j]);
}

double updateB(int i, int j, double ei, double ej, double ai_old, double aj_old) {
	double inProd_ii = 0, inProd_ij = 0, inProd_jj = 0;
	for (int d=0; d<xDim; d++) {
		inProd_ii += x[i][d] * x[i][d];
		inProd_ij += x[i][d] * x[j][d];
		inProd_jj += x[j][d] * x[j][d];
	}
	double b1 = b - ei - y[i]*(a[i]-ai_old)*inProd_ii - y[j]*(a[j]-aj_old)*inProd_ij;
	double b2 = b - ej - y[i]*(a[i]-ai_old)*inProd_ij - y[j]*(a[j]-aj_old)*inProd_jj;
	double finalB = 0;
	if (0 < a[i] && a[i] < C) finalB = b1;
	else if (0 < a[j] && a[j] < C) finalB = b2;
	else finalB = (b1 + b2) / 2;
	return finalB;
}

void train() {
	int pass = 0;
	int m = sampleNum * TRAIN_RATIO + 1;
	
	int iter=0;
    const int max_iter=50;
	while (pass < MAX_PASSES && iter < max_iter) {
		int num_changed_alphas = 0;
		printf("iter %d\n",iter);
		for (int i=0; i<m; i++) {
			double ei = calE(m, x[i], y[i]);
			if ((y[i]*ei < -EPS && a[i] < C) || (y[i]*ei > EPS && a[i] > 0)) {
				int j = rand() % m == i? (i+1)%m : rand()%m;
				double ej = calE(m, x[j], y[j]);
				double ai_old = a[i], aj_old = a[j];
				double lb = 0, rb = C;
				calLH(y[i], y[j], a[i], a[j], lb, rb);
				if (abs(lb - rb) < EPS)
					continue ;
				double ita = calIta(x[i], x[j]);
				if (ita >= 0)
					continue ;
				a[j] = computeAjAndClipValue(ei, ej, ita, lb, rb, j);
				if (abs(a[j] - aj_old) < 1e-5)
					continue ;
				a[i] = determineAi(i, j, aj_old);
				b = updateB(i, j, ei, ej, ai_old, aj_old);
				num_changed_alphas ++;
			}
		}
		if (num_changed_alphas == 0) pass ++;
		else pass = 0;
		++iter;
	}
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

int main() {
	init();
	input();
	train();
	eval();
	return 0;
}