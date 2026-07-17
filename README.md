# 1D-CNN Accelerator for PPG Signal Processing

## Overview
This repository contains the hardware implementation of a 1D Convolutional Neural Network (CNN) accelerator, specifically designed for Photoplethysmography (PPG) signal processing and heart rate monitoring. The architecture is developed with a strong focus on resource optimization and power efficiency for deployment on FPGA platforms.

## Project Structure
The repository is organized into the following core directories:

* **`C/`**: Contains the C++ implementation of the CNN layers and the main network architecture.
  * Core modules: `CNN.cpp`, `Conv.cpp`, `Dense.cpp`, `Pool.cpp`.
  * **`tb/`**: Testbench directory containing `CNN_tb.cpp` for simulation, along with input data (`X.txt`), expected output (`Y.txt`), and floating-point model weights (`Float_Weights.txt`).
* **`Python/`**: Contains Python utility scripts (`import.py`) for data preprocessing, model training, or weight extraction.
* **`rtl/`**: Contains the synthesized hardware description files generated for FPGA deployment.
  * **`verilog/`**: Verilog RTL implementation of the accelerator, including AXI memory-mapped interfaces (`gmem_m_axi`) and pipeline control logic.
  * **`vhdl/`**: VHDL RTL implementation of the accelerator.

## Key Features
* **Application-Specific Design:** Tailored specifically for 1D medical signal classification (PPG).
* **Hardware-Software Co-design:** Includes a full C++ reference model and testbench suite to verify hardware outputs against the software model.
* **Highly Optimized:** RTL generated via High-Level Synthesis (HLS) with explicit pipelining and loop unrolling (evident in `VITIS_LOOP` modules) to ensure minimal resource utilization and high power efficiency.
* **Standard Interfaces:** Utilizes AXI interfaces for seamless integration into larger SoC designs.

## Getting Started
### Prerequisites
* Xilinx Vivado / Vitis HLS (Version: 2022.2)
* C++ Compiler (GCC/G++) for testbench execution
* Python 3.12 for data preprocessing

### Simulation & Verification
1. Navigate to the `C/tb/` directory.
2. Compile the C++ files using the provided testbench.
3. Run the executable to verify the predicted outputs (`Y.txt`) against the input data (`X.txt`).

## Acknowledgments
Developed as part of a research initiative focusing on efficient AI hardware accelerators for biomedical applications.
Developed as part of a research initiative focusing on efficient AI hardware accelerators for biomedical applications.
