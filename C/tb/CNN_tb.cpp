#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include "CNN.h"

#define NumberOfPicture 3
#define INPUT_SIZE 1024

int main() {
    fxp OutModel0[1];
    fxp* Weights = (fxp*)malloc(49617 * sizeof(fxp));
    fxp* InModel = (fxp*)malloc((NumberOfPicture * INPUT_SIZE) * sizeof(fxp));
    fxp* Label = (fxp*)malloc(NumberOfPicture * sizeof(fxp));
    float tmp;

    FILE* WeightFile = fopen("Float_Weights.txt", "r");
    for (int i = 0; i < 49617; i++) { fscanf(WeightFile, "%f", &tmp); Weights[i] = (fxp)tmp; }
    fclose(WeightFile);

    FILE* InputFile = fopen("X.txt", "r");
    for (int i = 0; i < NumberOfPicture * INPUT_SIZE; i++) { fscanf(InputFile, "%f", &tmp); InModel[i] = (fxp)tmp; }
    fclose(InputFile);

    FILE* OutputFile = fopen("Y.txt", "r");
    for (int i = 0; i < NumberOfPicture; i++) { fscanf(OutputFile, "%f", &tmp); Label[i] = (fxp)tmp; }
    fclose(OutputFile);

    double totalError = 0;
    for (int i = 0; i < NumberOfPicture; i++) {
        fxp Image[1024];
        for (int k = 0; k < 1024; k++) Image[k] = InModel[i * 1024 + k];

        CNN(Image, OutModel0, Weights);

        float pred = (float)OutModel0[0];
        float real = (float)Label[i];
        totalError += fabsf(pred - real);
        printf("Sample %d: Real = %.2f, Pred = %.2f\n", i, real, pred);
    }
    printf("\nMAE: %.4f BPM\n", totalError / NumberOfPicture);
    free(Weights); free(InModel); free(Label);
    return 0;
}
