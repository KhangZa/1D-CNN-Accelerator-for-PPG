#include "Conv.h"
#include "Pool.h"
#include "Dense.h"
#include <algorithm>
#include <string.h>
#include <ap_axi_sdata.h>

void CNN(fxp InModel[1024], fxp OutModel0[1], fxp Weights[49617]){
    #pragma HLS INTERFACE m_axi port=InModel offset=slave bundle=gmem0 depth=1024
    #pragma HLS INTERFACE m_axi port=Weights offset=slave bundle=gmem1 depth=49617
    #pragma HLS INTERFACE s_axilite port=InModel bundle=control
    #pragma HLS INTERFACE s_axilite port=Weights bundle=control
    #pragma HLS INTERFACE s_axilite port=OutModel0 bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    fxp OutPadConv0[1056];
    fxp conv1d[8192];
    fxp re_lu[8192];
    fxp batch_normalization[8192];
    fxp average_pooling1d[2048];
    fxp OutPadConv1[2304];
    fxp conv1d_1[3072];
    fxp re_lu_1[3072];
    fxp batch_normalization_1[3072];
    fxp average_pooling1d_1[1536];
    fxp OutPadConv2[1920];
    fxp conv1d_2[2048];
    fxp re_lu_2[2048];
    fxp batch_normalization_2[2048];
    fxp average_pooling1d_2[1024];
    fxp straight_attention[1024];
    fxp global_average_pooling1d[64];
    fxp dense[32];

    memset(OutPadConv0, 0, sizeof(OutPadConv0));
    memset(conv1d, 0, sizeof(conv1d));
    memset(re_lu, 0, sizeof(re_lu));
    memset(batch_normalization, 0, sizeof(batch_normalization));
    memset(average_pooling1d, 0, sizeof(average_pooling1d));
    memset(OutPadConv1, 0, sizeof(OutPadConv1));
    memset(conv1d_1, 0, sizeof(conv1d_1));
    memset(re_lu_1, 0, sizeof(re_lu_1));
    memset(batch_normalization_1, 0, sizeof(batch_normalization_1));
    memset(average_pooling1d_1, 0, sizeof(average_pooling1d_1));
    memset(OutPadConv2, 0, sizeof(OutPadConv2));
    memset(conv1d_2, 0, sizeof(conv1d_2));
    memset(re_lu_2, 0, sizeof(re_lu_2));
    memset(batch_normalization_2, 0, sizeof(batch_normalization_2));
    memset(average_pooling1d_2, 0, sizeof(average_pooling1d_2));
    memset(straight_attention, 0, sizeof(straight_attention));
    memset(global_average_pooling1d, 0, sizeof(global_average_pooling1d));
    memset(dense, 0, sizeof(dense));

    Padding_Conv1D_0(&InModel[0], OutPadConv0);
    Conv1D_0(OutPadConv0, conv1d, &Weights[1152], &Weights[0]);
    Activation0(conv1d, re_lu);
    BatchNorm1D_0(re_lu, batch_normalization, &Weights[1184], &Weights[1216], &Weights[1248], &Weights[1280]);
    Average_Pool1D_0(batch_normalization, average_pooling1d);
    
    Padding_Conv1D_1(average_pooling1d, OutPadConv1);
    Conv1D_1(OutPadConv1, conv1d_1, &Weights[15136], &Weights[1312]);
    Activation1(conv1d_1, re_lu_1);
    BatchNorm1D_1(re_lu_1, batch_normalization_1, &Weights[15184], &Weights[15232], &Weights[15280], &Weights[15328]);
    Average_Pool1D_1(batch_normalization_1, average_pooling1d_1);
    
    Padding_Conv1D_2(average_pooling1d_1, OutPadConv2);
    Conv1D_2(OutPadConv2, conv1d_2, &Weights[43024], &Weights[15376]);
    Activation2(conv1d_2, re_lu_2);
    BatchNorm1D_2(re_lu_2, batch_normalization_2, &Weights[43088], &Weights[43152], &Weights[43216], &Weights[43280]);
    Average_Pool1D_2(batch_normalization_2, average_pooling1d_2);
    
    Conv1D_3(average_pooling1d_2, straight_attention, &Weights[47440], &Weights[43344]);
    GlobalAveragePool1D_0(straight_attention, global_average_pooling1d);
    Dense_0(global_average_pooling1d, dense, &Weights[49552], &Weights[47504]);
    Dense_1(dense, OutModel0, &Weights[49616], &Weights[49584]);
}
