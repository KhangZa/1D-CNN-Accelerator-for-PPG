#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
//typedef float fxp;
void Average_Pool1D_0(fxp input_AveragePooling[8192], fxp output_AveragePooling[2048]){
	int PoolSize = 4;
	int stride= 4;
	int index = 0;
	loop_for_channel_pool_0:
	for (int z = 0; z < 32; z++){
		index = 0;
		loop_for_weight_pool_0:
		for (int y = 0; y < 64; y++){
			fxp Average = 0;
			fxp pool_value = 0;
			for (int j = 0; j < PoolSize; j++)
			{
				int pool_index = 256 * z + j + y * stride;
				pool_value += input_AveragePooling[pool_index];
			}
			Average = pool_value / PoolSize;
			int out_index = 64 * z + index;
			output_AveragePooling[out_index]=Average;
			index++;
		}
	}
}
void Average_Pool1D_1(fxp input_AveragePooling[3072], fxp output_AveragePooling[1536]){
	int PoolSize = 2;
	int stride= 2;
	int index = 0;
	loop_for_channel_pool_1:
	for (int z = 0; z < 48; z++){
		index = 0;
		loop_for_weight_pool_1:
		for (int y = 0; y < 32; y++){
			fxp Average = 0;
			fxp pool_value = 0;
			for (int j = 0; j < PoolSize; j++)
			{
				int pool_index = 64 * z + j + y * stride;
				pool_value += input_AveragePooling[pool_index];
			}
			Average = pool_value / PoolSize;
			int out_index = 32 * z + index;
			output_AveragePooling[out_index]=Average;
			index++;
		}
	}
}
void Average_Pool1D_2(fxp input_AveragePooling[2048], fxp output_AveragePooling[1024]){
	int PoolSize = 2;
	int stride= 2;
	int index = 0;
	loop_for_channel_pool_2:
	for (int z = 0; z < 64; z++){
		index = 0;
		loop_for_weight_pool_2:
		for (int y = 0; y < 16; y++){
			fxp Average = 0;
			fxp pool_value = 0;
			for (int j = 0; j < PoolSize; j++)
			{
				int pool_index = 32 * z + j + y * stride;
				pool_value += input_AveragePooling[pool_index];
			}
			Average = pool_value / PoolSize;
			int out_index = 16 * z + index;
			output_AveragePooling[out_index]=Average;
			index++;
		}
	}
}
void GlobalAveragePool1D_0(fxp input_GlobalAveragePool1D[1024],fxp output_GlobalAveragePool1D[64]){
	int hs = 0;
	for (int i = 0; i < 64; i++){
		fxp avg = 0;
		for (int j = 0; j < 16; j++){
			avg += input_GlobalAveragePool1D[16 * i + j] / 16;
		}
		output_GlobalAveragePool1D[hs] = avg;
		hs++;
	}
}
