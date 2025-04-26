# RDP++ (Recombinant Detection Prediction)

A C++ application for recombinant detection prediction using neural networks with ONNX Runtime.

## Overview

RDP++ is a tool that uses a trained neural network model to predict recombinant classes based on a set of input features. The system consists of two main components:

1. **PSNN (Prediction System Neural Network)**: Processes standardized input data and runs inference using an ONNX model.
2. **Tester**: Prepares test data and interacts with PSNN to generate predictions.

The system reads feature data from a shared file, processes it, and outputs classification probabilities for different recombinant classes.

## Requirements

- C++17 compatible compiler (g++, MSVC, etc.)
- ONNX Runtime (version 1.21.1 or compatible)
- For GPU acceleration: CUDA compatible GPU with appropriate drivers

## Project Structure

- `PSNN.cpp` and `PSNN.h`: Main prediction system that processes input data and runs the neural network model
- `tester.cpp`: Tool for generating test data and running the prediction system
- `RDP_TripleNN.onnx`: The trained neural network model
- `sharedData.txt`: Input data file shared between components
- `prediction_result.txt`: Output file containing prediction results

## Building the Project

### Linux

```bash
# Compile PSNN
g++ -std=c++17 PSNN.cpp -o PSNN -I./onnxruntime-linux-x64-gpu-1.21.1/include -L./onnxruntime-linux-x64-gpu-1.21.1/lib -lonnxruntime

# Compile tester
g++ -std=c++17 tester.cpp -o tester
```

### Windows

```bash
# Compile PSNN
cl /std:c++17 PSNN.cpp /Fe:PSNN.exe /I"path\to\onnxruntime\include" /link "path\to\onnxruntime\lib\onnxruntime.lib"

# Compile tester
cl /std:c++17 tester.cpp /Fe:tester.exe
```

## Usage

### Running a Prediction

1. Create input data in `sharedData.txt` or use the tester to generate sample data
2. Run the prediction:

```bash
# Linux
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./onnxruntime-linux-x64-gpu-1.21.1/lib
./tester

# Windows
.\tester.exe
```

### Input Data Format

The input data file (`sharedData.txt`) uses a comma-separated format:
```
FeatureName1,FeatureValue1
FeatureName2,FeatureValue2
...
```

### Output Format

The prediction results are stored in `prediction_result.txt`:
```
Class 0,Class 1,Class 2,
0.628235,0.326715,0.0450506,
Predicted: Class 0 with 62.8235% confidence
```

## Integration with Other Applications

The system is designed to be integrated with other applications through file-based communication:

1. External application generates input data in `sharedData.txt`
2. External application runs `tester` (or uses the functionality in `tester.cpp`)
3. External application reads prediction results from `prediction_result.txt`

## Model Details

The prediction model (`RDP_TripleNN.onnx`) is a neural network that classifies inputs into three recombinant classes. The model expects standardized input features.

## Data Processing Pipeline

1. Read raw input data from `sharedData.txt`
2. Drop features with no variance
3. Standardize features using pre-calculated means and standard deviations
4. Run inference through the ONNX model
5. Generate classification results and save to `prediction_result.txt`

## License

MIT

## Acknowledgements

- ONNX Runtime for providing the neural network execution environment# RdpLink
