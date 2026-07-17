#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
//typedef float fxp;
void Average_Pool1D_0(fxp input_AveragePooling[8192], fxp output_AveragePooling[2048]);
void Average_Pool1D_1(fxp input_AveragePooling[3072], fxp output_AveragePooling[1536]);
void Average_Pool1D_2(fxp input_AveragePooling[2048], fxp output_AveragePooling[1024]);
void GlobalAveragePool1D_0(fxp input_GlobalAveragePool1D[1024],fxp output_GlobalAveragePool1D[64]);
