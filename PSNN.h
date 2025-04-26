#ifndef PSNN_H
#define PSNN_H

#include <vector>
#include <string>

/**
 * Remove elements from the vector that do not show variance in the python script.
 * 
 * @param names Vector of feature names
 * @param scores Vector of feature scores
 * @return 0 on success, non-zero on failure
 */
int drop(std::vector<std::string>& names, std::vector<double>& scores);

/**
 * Standardize the remaining scores using means and standard deviations.
 * 
 * @param names Vector of feature names
 * @param scores Vector of feature scores to be standardized
 * @return 0 on success, non-zero on failure
 */
int standardise(std::vector<std::string>& names, std::vector<double>& scores);

/**
 * Run inference on the input data using the ONNX model.
 * 
 * @param input_name Vector of feature names
 * @param input_tensor_values Vector of feature values to use for inference
 * @return 0 on success, non-zero on failure
 */
int inference(std::vector<std::string>& input_name, std::vector<double>& input_tensor_values);

#endif // PSNN_H