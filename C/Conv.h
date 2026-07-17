#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
//typedef float fxp;
void Padding_Conv1D_0(fxp input_Pad_Conv[1024], fxp output_Pad_Conv[1056]);
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
//typedef float fxp;
void Conv1D_0(fxp Input_Conv[1056],fxp Output_Conv[8192], fxp bias[32], fxp kernel[1152]);
void Activation0(fxp Input_Activation[8192], fxp Output_Activation[8192]);
void BatchNorm1D_0(fxp Input_BatchNorm[8192], fxp Output_BatchNorm[8192], fxp gamma[32], fxp beta[32], fxp MovMean[32], fxp MovVar[32]);
void Padding_Conv1D_1(fxp input_Pad_Conv[2048], fxp output_Pad_Conv[2304]);
void Conv1D_1(fxp Input_Conv[2304],fxp Output_Conv[3072], fxp bias[48], fxp kernel[13824]);
void Activation1(fxp Input_Activation[3072], fxp Output_Activation[3072]);
void BatchNorm1D_1(fxp Input_BatchNorm[3072], fxp Output_BatchNorm[3072], fxp gamma[48], fxp beta[48], fxp MovMean[48], fxp MovVar[48]);
void Padding_Conv1D_2(fxp input_Pad_Conv[1536], fxp output_Pad_Conv[1920]);
void Conv1D_2(fxp Input_Conv[1920],fxp Output_Conv[2048], fxp bias[64], fxp kernel[27648]);
void Activation2(fxp Input_Activation[2048], fxp Output_Activation[2048]);
void BatchNorm1D_2(fxp Input_BatchNorm[2048], fxp Output_BatchNorm[2048], fxp gamma[64], fxp beta[64], fxp MovMean[64], fxp MovVar[64]);
void Conv1D_3(fxp Input_Conv[1024],fxp Output_Conv[1024], fxp bias[64], fxp kernel[4096]);
