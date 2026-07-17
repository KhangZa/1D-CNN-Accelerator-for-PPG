#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
//typedef float fxp;
void Dense_0(fxp input_Dense[64],fxp output_Dense[32],fxp bias[32],fxp weight[2048]);
void Dense_1(fxp input_Dense[32],fxp output_Dense[1],fxp bias[1],fxp weight[32]);
