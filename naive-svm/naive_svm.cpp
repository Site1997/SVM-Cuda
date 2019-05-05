#include <queue>
#include <cmath>
#include <cstdio>
#include <vector>
#include <cstring>
#include <iostream>
#include <algorithm>
using namespace std;

const int MAXN = 500;
const int MAX_EPOCH = 10000;
const double TRAIN_RATIO = 0.2;
const double ALPHA = 0.0001;

int sampleNum, featureDim;
double feature[MAXN][MAXN]; int label[MAXN];
double weights[MAXN];

void init() {
	sampleNum = featureDim = 0;
	memset(weights, 0, sizeof(weights));
	memset(feature, 0, sizeof(feature));
	memset(label, 0, sizeof(label));
}

void input() {
	freopen("in.txt", "r", stdin);
	scanf("%d%d", &sampleNum, &featureDim);
	for (int i=0; i<sampleNum; i++) {
		for (int j=0; j<featureDim; j++)
			scanf("%lf", &feature[i][j]);
		scanf("%d", &label[i]);
		// printf("%.2f %.2f %d\n", feature[i][0], feature[i][1], label[i]);
	}
}

void train() {
	int trainNum = sampleNum * TRAIN_RATIO + 1;
	double y[MAXN], prod[MAXN], update[MAXN];
	for (int e=1; e<=MAX_EPOCH; e++) {
		// calculate y and prod
		for (int i=0; i<trainNum; i++) {
			double res = 0;
			for (int d=0; d<featureDim; d++) {
				res += weights[d] * feature[i][d];
			}
			y[i] = res;
			prod[i] = y[i] * label[i];
		}
		// init "update"
		memset(update, 0, sizeof(update));
		for (int d=0; d<featureDim; d++) {
			update[d] = - (2.0 * (1.0/(double)e) * weights[d]);
		}
		// accumulate "update"
		for (int i=0; i<trainNum; i++) {
			if (prod[i] < 1) {
				for (int d=0; d<featureDim; d++)
					update[d] += feature[i][d] * label[i];
			}
		}
		// update weights
		for (int d=0; d<featureDim; d++) {
			weights[d] += ALPHA * update[d];
		}
	}
}

void eval() {
	double y_pred[MAXN];
	int correctNum = 0;
	int trainNum = sampleNum * TRAIN_RATIO + 1;
	for (int i=trainNum; i<sampleNum; i++) {
		double res = 0;
		for (int d=0; d<featureDim; d++)
			res += weights[d] * feature[i][d];
		int curPred = 0;
		if (res > 1) curPred = 1;
		else curPred = -1;
		correctNum += (curPred == label[i]);
	}
	printf("Final Accuracy : %.4f\n", (double)correctNum / (double)(sampleNum-trainNum));
}

int main() {
	init();
	input();
	train();
	eval();
	return 0;
}