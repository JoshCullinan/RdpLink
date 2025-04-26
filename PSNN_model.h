#ifndef PSNN_MODEL_H
#define PSNN_MODEL_H

#include <vector>
#include <string>

/**
 * Inference class for ONNX model predictions
 */
class PSNNModel {
public:
    /**
     * Constructor
     * 
     * @param model_path Path to the ONNX model file
     */
    PSNNModel(const std::string& model_path);
    
    /**
     * Destructor
     */
    ~PSNNModel();
    
    /**
     * Run inference on input data
     * 
     * @param input_values Vector of input values
     * @param class_probabilities Output vector to store class probabilities
     * @return 0 on success, non-zero on failure
     */
    int predict(const std::vector<float>& input_values, std::vector<float>& class_probabilities);
    
    /**
     * Get the index of the highest probability class
     * 
     * @param class_probabilities Vector of class probabilities
     * @return Index of the highest probability class
     */
    static size_t getMaxProbabilityClass(const std::vector<float>& class_probabilities);

private:
    // Implementation details
    class Impl;
    Impl* pImpl;
};

#endif // PSNN_MODEL_H