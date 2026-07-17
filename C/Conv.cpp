#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
//typedef float fxp;
void Padding_Conv1D_0(fxp input_Pad_Conv[1024], fxp output_Pad_Conv[1056]){
	loop_for_3_channel_pad_0:
	for (int c = 0; c < 4; c++){
		loop_for_channel_pad_0:
		for (int n = 0; n < 264; n++){
			if (n < 4 || n >= 260) output_Pad_Conv[264 * c + n]=0; else output_Pad_Conv[264 * c + n] = input_Pad_Conv[256 * c + n - 4];
		}
	}
}
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
//typedef float fxp;
void Conv1D_0(fxp Input_Conv[1056],fxp Output_Conv[8192], fxp bias[32], fxp kernel[1152]){
	loop_for_channel_0:
	int stride = 1;
	for (int n = 0; n < 32; n++){
		loop_for_ap_0:
		for (int y = 0; y < 256; y++){
			fxp s = 0;
			loop_for_fc_0:
			for (int k = 0; k < 4; k++){
				loop_for_fa_0:
				for (int j = 0; j < 9; j++){
					s=s+(kernel[4*9*n+9*k+j])*(Input_Conv[264*k+j+y*stride]);}
			}
			Output_Conv[256*n+y]=s+bias[n];
		}
	}
}
 void Activation0(fxp Input_Activation[8192], fxp Output_Activation[8192]){
	for (int i = 0; i < 8192; i++){
		if(Input_Activation[i] > 0){
			Output_Activation[i] = Input_Activation[i];
		}else
		{
			Output_Activation[i] = 0;
		}
	}
}
#include <hls_math.h>
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
//typedef float fxp;
 void BatchNorm1D_0(fxp Input_BatchNorm[8192], fxp Output_BatchNorm[8192], fxp gamma[32],fxp beta[32], fxp MovMean[32], fxp MovVar[32]) {
	fxp eps = 0.001;
	 for(int i = 0; i < 32; i++){
		for(int j = 0; j < 256; j++){
			Output_BatchNorm[256 * i + j] = ((Input_BatchNorm[256 * i + j] - MovMean[i]) / (hls::sqrt(MovVar[i] + eps)))* gamma[i] + beta[i];
		}
	}
}
void Padding_Conv1D_1(fxp input_Pad_Conv[2048], fxp output_Pad_Conv[2304]){
	loop_for_3_channel_pad_1:
	for (int c = 0; c < 32; c++){
		loop_for_channel_pad_1:
		for (int n = 0; n < 72; n++){
			if (n < 4 || n >= 68) output_Pad_Conv[72 * c + n]=0; else output_Pad_Conv[72 * c + n] = input_Pad_Conv[64 * c + n - 4];
		}
	}
}
void Conv1D_1(fxp Input_Conv[2304],fxp Output_Conv[3072], fxp bias[48], fxp kernel[13824]){
	loop_for_channel_1:
	int stride = 1;
	for (int n = 0; n < 48; n++){
		loop_for_ap_1:
		for (int y = 0; y < 64; y++){
			fxp s = 0;
			loop_for_fc_1:
			for (int k = 0; k < 32; k++){
				loop_for_fa_1:
				for (int j = 0; j < 9; j++){
					s=s+(kernel[32*9*n+9*k+j])*(Input_Conv[72*k+j+y*stride]);}
			}
			Output_Conv[64*n+y]=s+bias[n];
		}
	}
}
 void Activation1(fxp Input_Activation[3072], fxp Output_Activation[3072]){
	for (int i = 0; i < 3072; i++){
		if(Input_Activation[i] > 0){
			Output_Activation[i] = Input_Activation[i];
		}else
		{
			Output_Activation[i] = 0;
		}
	}
}
#include <hls_math.h>
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
 //typedef float fxp;
 void BatchNorm1D_1(fxp Input_BatchNorm[3072], fxp Output_BatchNorm[3072], fxp gamma[48],fxp beta[48], fxp MovMean[48], fxp MovVar[48]) {
	fxp eps = 0.001;
	 for(int i = 0; i < 48; i++){
		for(int j = 0; j < 64; j++){
			Output_BatchNorm[64 * i + j] = ((Input_BatchNorm[64 * i + j] - MovMean[i]) / (hls::sqrt(MovVar[i] + eps)))* gamma[i] + beta[i];
		}
	}
}
void Padding_Conv1D_2(fxp input_Pad_Conv[1536], fxp output_Pad_Conv[1920]){
	loop_for_3_channel_pad_2:
	for (int c = 0; c < 48; c++){
		loop_for_channel_pad_2:
		for (int n = 0; n < 40; n++){
			if (n < 4 || n >= 36) output_Pad_Conv[40 * c + n]=0; else output_Pad_Conv[40 * c + n] = input_Pad_Conv[32 * c + n - 4];
		}
	}
}
void Conv1D_2(fxp Input_Conv[1920],fxp Output_Conv[2048], fxp bias[64], fxp kernel[27648]){
	loop_for_channel_2:
	int stride = 1;
	for (int n = 0; n < 64; n++){
		loop_for_ap_2:
		for (int y = 0; y < 32; y++){
			fxp s = 0;
			loop_for_fc_2:
			for (int k = 0; k < 48; k++){
				loop_for_fa_2:
				for (int j = 0; j < 9; j++){
					s=s+(kernel[48*9*n+9*k+j])*(Input_Conv[40*k+j+y*stride]);}
			}
			Output_Conv[32*n+y]=s+bias[n];
		}
	}
}
 void Activation2(fxp Input_Activation[2048], fxp Output_Activation[2048]){
	for (int i = 0; i < 2048; i++){
		if(Input_Activation[i] > 0){
			Output_Activation[i] = Input_Activation[i];
		}else
		{
			Output_Activation[i] = 0;
		}
	}
}
#include <hls_math.h>
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
 //typedef float fxp;
 void BatchNorm1D_2(fxp Input_BatchNorm[2048], fxp Output_BatchNorm[2048], fxp gamma[64],fxp beta[64], fxp MovMean[64], fxp MovVar[64]) {
	fxp eps = 0.001;
	 for(int i = 0; i < 64; i++){
		for(int j = 0; j < 32; j++){
			Output_BatchNorm[32 * i + j] = ((Input_BatchNorm[32 * i + j] - MovMean[i]) / (hls::sqrt(MovVar[i] + eps)))* gamma[i] + beta[i];
		}
	}
}
void Conv1D_3(fxp Input_Conv[1024],fxp Output_Conv[1024], fxp bias[64], fxp kernel[4096]){
	loop_for_channel_3:
	int stride = 1;
	for (int n = 0; n < 64; n++){
		loop_for_ap_3:
		for (int y = 0; y < 16; y++){
			fxp s = 0;
			loop_for_fc_3:
			for (int k = 0; k < 64; k++){
				loop_for_fa_3:
				for (int j = 0; j < 1; j++){
					s=s+(kernel[64*1*n+1*k+j])*(Input_Conv[16*k+j+y*stride]);}
			}
			if ((s+bias[n])<0) Output_Conv[16*n+y]=0; else Output_Conv[16*n+y]=s+bias[n];
		}
	}
}
