# 🫀 1D-CNN Hardware Accelerator for PPG

![Verilog/VHDL](https://img.shields.io/badge/Hardware-Verilog%20%2F%20VHDL-11557c?style=for-the-badge)
![C++](https://img.shields.io/badge/Software-C%2B%2B-00599C?style=for-the-badge)
![Python](https://img.shields.io/badge/Utility-Python-3776AB?style=for-the-badge)
![FPGA](https://img.shields.io/badge/Platform-Xilinx_FPGA-232F3E?style=for-the-badge)

## 📌 Overview
This repository contains the RTL implementation and software co-design of a **1D Convolutional Neural Network (CNN) Hardware Accelerator**, specifically tailored for Photoplethysmography (PPG) signal processing and heart rate monitoring. The system is designed to offload deep learning inference tasks to a dedicated FPGA IP core, focusing heavily on resource optimization and power efficiency for medical edge devices.

## ✨ Key Features
- **Application-Specific AI:** Hardware-accelerated 1D-CNN optimized for biomedical signal classification (PPG).
- **HLS & RTL Integration:** Synthesized using High-Level Synthesis (HLS) with explicit pipelining and loop unrolling for minimal latency, providing both Verilog and VHDL implementations.
- **Hardware-Software Co-verification:** The hardware output is rigorously verified against a standard C++ reference model using a comprehensive testbench.
- **Standard SoC Interfaces:** Utilizes AXI memory-mapped interfaces (`gmem_m_axi`) for seamless integration into larger SoC architectures.

## 🛠️ Tools & Technologies
- **Hardware Description:** Verilog HDL, VHDL
- **Software Verification & Modeling:** C++, Python
- **EDA/Synthesis Tools:** Xilinx Vivado / Vitis HLS
- **Target Platform:** Xilinx FPGA

## 🚀 Getting Started

### Prerequisites
Make sure you have Xilinx Vivado / Vitis HLS installed, along with a C++ compiler (GCC) and Python 3.x for testbench execution and data preprocessing.

### Running the Project
1. Clone this repository:
   ```bash
   git clone [https://github.com/KhangZa/1d-cnn-accelerator-for-ppg.git](https://github.com/KhangZa/1d-cnn-accelerator-for-ppg.git)
