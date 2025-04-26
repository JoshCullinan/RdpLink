// rdp_example.cpp - Example of how RDP would use PSNN.dll

#include <windows.h>
#include <iostream>
#include <vector>
#include <string>
#include "PSNN.h"

// Function pointer types for dynamic loading
typedef bool (*PSNN_Initialize_Func)(const char*);
typedef bool (*PSNN_Predict_Func)(const char**, const double*, int, PredictionResult*);
typedef void (*PSNN_Cleanup_Func)();

class PSNNIntegration {
private:
    HMODULE dllHandle;
    PSNN_Initialize_Func initialize;
    PSNN_Predict_Func predict;
    PSNN_Cleanup_Func cleanup;
    
public:
    PSNNIntegration() : dllHandle(nullptr), initialize(nullptr), predict(nullptr), cleanup(nullptr) {}
    
    ~PSNNIntegration() {
        unload();
    }
    
    // Load the DLL and get function pointers
    bool load() {
        // Load the DLL
        dllHandle = LoadLibrary("PSNN.dll");
        if (!dllHandle) {
            std::cerr << "Failed to load PSNN.dll. Error: " << GetLastError() << std::endl;
            return false;
        }
        
        // Get function pointers
        initialize = (PSNN_Initialize_Func)GetProcAddress(dllHandle, "PSNN_Initialize");
        predict = (PSNN_Predict_Func)GetProcAddress(dllHandle, "PSNN_Predict");
        cleanup = (PSNN_Cleanup_Func)GetProcAddress(dllHandle, "PSNN_Cleanup");
        
        if (!initialize || !predict || !cleanup) {
            std::cerr << "Failed to get function pointers from PSNN.dll" << std::endl;
            unload();
            return false;
        }
        
        return true;
    }
    
    // Initialize the model
    bool initializeModel(const char* modelPath) {
        if (!initialize) return false;
        return initialize(modelPath);
    }
    
    // Make a prediction
    bool makePrediction(const std::vector<std::string>& names, 
                       const std::vector<double>& values, 
                       PredictionResult& result) {
        if (!predict || names.size() != values.size()) return false;
        
        // Convert to C-style arrays
        std::vector<const char*> name_ptrs;
        for (const auto& name : names) {
            name_ptrs.push_back(name.c_str());
        }
        
        return predict(name_ptrs.data(), values.data(), names.size(), &result);
    }
    
    // Cleanup and unload
    void unload() {
        if (cleanup) {
            cleanup();
        }
        
        if (dllHandle) {
            FreeLibrary(dllHandle);
            dllHandle = nullptr;
        }
        
        initialize = nullptr;
        predict = nullptr;
        cleanup = nullptr;
    }
};

// Example usage in RDP
int main() {
    PSNNIntegration psnn;
    
    // Load the DLL
    if (!psnn.load()) {
        std::cerr << "Failed to load PSNN DLL" << std::endl;
        return 1;
    }
    
    // Initialize with model path
    const char* modelPath = "C:\\path\\to\\RDP_TripleNN.onnx";
    if (!psnn.initializeModel(modelPath)) {
        std::cerr << "Failed to initialize PSNN model" << std::endl;
        return 1;
    }
    
    // Prepare input data (this would come from RDP's calculations)
    std::vector<std::string> names = {
        "ListCorr(A)1", "ListCorr(A)2", "ListCorr(A)3",
        "SimScoreB(A)1", "SimScoreB(A)2", "SimScoreB(A)3",
        // ... all 117 feature names
    };
    
    std::vector<double> values = {
        19.0, 30.0, 20.0,
        0.22937, 0.01384, -0.01384,
        // ... all 117 values
    };
    
    // Make prediction
    PredictionResult result;
    if (psnn.makePrediction(names, values, result)) {
        std::cout << "Prediction successful!" << std::endl;
        std::cout << "Predicted class: " << result.predicted_class << std::endl;
        std::cout << "Confidence: " << result.confidence * 100.0f << "%" << std::endl;
        std::cout << "Class probabilities:" << std::endl;
        for (int i = 0; i < 3; i++) {
            std::cout << "  Class " << i << ": " << result.class_probabilities[i] * 100.0f << "%" << std::endl;
        }
    } else {
        std::cerr << "Prediction failed" << std::endl;
    }
    
    // Cleanup is automatic when psnn goes out of scope
    return 0;
}