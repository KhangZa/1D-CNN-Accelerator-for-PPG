#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
//typedef float fxp;
void Dense_0(fxp input_Dense[64],fxp output_Dense[32],fxp bias[32],fxp weight[2048]){
	fxp out_Dense[32];
	loop_for_a_Dense_0:
	for (int i = 0; i < 32; i++){
		fxp s=0;
		loop_for_b_Dense_0:
		for (int j = 0; j < 64; j++){
			s+=input_Dense[j]*weight[j*32+i];
		}
		out_Dense[i]=s+bias[i];
	}
	for (int i = 0; i < 32; i++){
		if (out_Dense[i] < 0) output_Dense[i] = 0; else output_Dense[i] = out_Dense[i];
	}
}
void Dense_1(fxp input_Dense[32],fxp output_Dense[1],fxp bias[1],fxp weight[32]){
	fxp out_Dense[1];
	loop_for_a_Dense_1:
	for (int i = 0; i < 1; i++){
		fxp s=0;
		loop_for_b_Dense_1:
		for (int j = 0; j < 32; j++){
			s+=input_Dense[j]*weight[j*1+i];
		}
		out_Dense[i]=s+bias[i];
	}
	for (int i = 0; i < 1; i++){
		output_Dense[i] = out_Dense[i];
	}
}
