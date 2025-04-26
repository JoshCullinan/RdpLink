// PSNN.h - Header file for PSNN DLL integration
#ifndef PSNN_DLL_H
#define PSNN_DLL_H

// Windows-specific DLL export/import macros
#ifdef _WIN32
    #ifdef PSNN_EXPORTS
        #define PSNN_API __declspec(dllexport)
    #else
        #define PSNN_API __declspec(dllimport)
    #endif
#else
    #define PSNN_API
#endif

// Structure for returning prediction results
struct PredictionResult {
    float class_probabilities[3];  // Probabilities for each class (0, 1, 2)
    int predicted_class;           // Class with highest probability
    float confidence;              // Confidence (probability) of predicted class
};

// C interface for compatibility
#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize the PSNN model
 * Must be called before using any other functions
 * 
 * @param model_path Full path to the ONNX model file (RDP_TripleNN.onnx)
 * @return true if successful, false otherwise
 */
PSNN_API bool PSNN_Initialize(const char* model_path);

/**
 * Process input data and return predictions
 * 
 * @param names Array of feature names (must match expected features)
 * @param values Array of feature values corresponding to the names
 * @param num_features Number of features in the arrays (should be 117)
 * @param result Pointer to PredictionResult structure to receive output
 * @return true if successful, false otherwise
 */
PSNN_API bool PSNN_Predict(const char** names, const double* values, int num_features, PredictionResult* result);

/**
 * Cleanup resources
 * Should be called when done using the DLL
 */
PSNN_API void PSNN_Cleanup();

#ifdef __cplusplus
}
#endif

#endif // PSNN_DLL_H