#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <onnxruntime_cxx_api.h>

#include "PSNN.h"

//Drop the elements from the vector that do not show variance in the pyton script.
int drop(std::vector<std::string>& names, std::vector<double>& scores) {

    std::vector<std::string> dropNames{"SRCompatF(A)1","SRCompatF(A)2","SRCompatF(A)3",
        "SRCompatS(A)1","SRCompatS(A)2", "SRCompatS(A)3",
        "RCompatXF(A)1","RCompatXF(A)2","RCompatXF(A)3",
        "RCompatXS(A)1", "RCompatXS(A)2","RCompatXS(A)3",
        "SetTot(1:A)1","SetTot(1:A)2","SetTot(1:A)3", "Consensus(A:0)1", "Consensus(A:0)2",
        "Consensus(A:0)3", "Consensus(A:1)1", "Consensus(A:1)2", "Consensus(A:1)3",
        "Consensus(A:2)1", "Consensus(A:2)2", "Consensus(A:2)3"
    };

    // Remove elements from names and scores if the name is in dropNames
    for (size_t i = 0; i < names.size(); ) {
        if (std::find(dropNames.begin(), dropNames.end(), names[i]) != dropNames.end()) {
            names.erase(names.begin() + i);
            scores.erase(scores.begin() + i);
        } else {
            ++i;
        }
    }
        
    return 0;
}


//Standardise the remaining scores using the means and standard deviations from the python script.
int standardise(std::vector<std::string>& names, std::vector<double>& scores){
    std::vector<double> stdDev{77.81935754, 78.61500891, 78.49932003, 0.33135391, 0.3302249,
        0.33013326,  0.47501296,  0.45518937,  0.45380344,  0.43516053,
        0.41983258, 0.42051523, 0.42452919, 0.40729272, 0.40618444,
        0.42237544, 0.40565562, 0.40395664, 9.18215223, 8.87685838,
        8.79940741, 0.29372744, 0.29004331, 0.28246039, 0.45595758,
        0.44363619, 0.44087034, 0.27690231, 0.27243035, 0.27253431,
        2.30955365, 2.26208653, 2.25805451, 0.29305205, 0.29300839,
        0.29317703, 2.62412539, 2.66208001, 2.75144994, 0.23280591,
        0.23614036, 0.23754669, 0.71210972, 0.73173777, 0.76329215,
        0.10588893, 0.10604115, 0.1073419, 2.5910722, 2.63099812,
        2.71640149, 0.22258118, 0.22258118, 0.22258118, 0.7824096,
        0.81190509, 0.85185175, 0.09563661, 0.09563661, 0.09563661,
        0.98819725, 1.04309565, 1.06479416, 1.00704312, 1.05179805,
        1.07383432, 9.91944706, 9.84952037, 9.67515909, 1.59625999,
        1.606194, 1.695653, 0.79266336, 0.82650908, 0.82976458,
        0.22446628, 0.22453972, 0.22425348, 0.12584292, 0.12653409,
        0.12659368, 56.66741131, 56.56714733, 56.6054163, 5.2931978,
        5.51612525, 5.7820158, 47.38165939, 47.11818704, 47.63537705,
       47.81972043, 47.4848627, 47.60851212,  0.21678955,  0.21445225,0.21430079 };

    std::vector<double> means{93.64829657377209,94.9787838666968,94.9850292246585,0.1652282657021991,0.13167896018342107,
        0.1261814410178577,0.34391448961798043,0.29311202247553847,0.290089450059741,0.4035286001872961,0.4692451581360804,
        0.4790747803145283,0.34072728498078597,0.39561142345077027,0.4007465964413731,0.3378896775276908,0.3926582591791263,
        0.39667262815254944,8.534188397590984,8.138416175767754,8.035000038001742,0.14400886621241968,0.13566300113023544,0.1323365148060839,
        0.29481060483740756,0.2693770788258469,0.2641327865146769,0.49313046145897244,0.45780350050053287,0.4528006135563666,1.9242273064875508,
        1.8578602447766979,1.8517204830949072,0.41068983543772397,0.38140402331514195,0.3787053476927051,2.291025930829593,2.5652985436109406,
        2.6910582232699327,0.018529402266929312,0.019362547227693996,0.01950463396518875,0.0891335938256854,0.09324119223689734,0.09845319210772759,
        0.002434850001614622,0.0024413084896825654,0.0024736009300222817,2.297264830303226,2.5751089869861468,2.704898763199535,0.01568766751703426,
        0.01568766751703426,0.01568766751703426,0.11023347434365614,0.11721509994510285,0.12358316918009495,0.0023056802402557563,0.0023056802402557563,
        0.0023056802402557563,0.2794264862595666,0.30316788839732617,0.31430878031452836,0.29001840669099366,0.31255853004811573,0.3237575483579294,
        7.34176035643104,7.0250873698194845,6.9248489843833765,1.1292924726321567,1.2199631866180127,1.3192043142700294,1.0061291051764782,0.9959505279813996,
        0.9979203668421223,0.3967839035747731,0.3919100010979429,0.39125673433009334,0.18867358949849838,0.1918002263700068,0.1927855593373591,-11.044021054671102,
        -13.168644040430136,-13.399354151193206,3.154015564956244,3.225704782510414,3.483837633609972,61.81170923886718,60.48893338069558,60.484057222204285,
        60.04446023185972,58.992721283947425,58.86758161914296,0.528549646892492,0.5111502837005976,0.5075989531249846};
    
    // Check if we have enough standardization parameters
    if (names.size() > stdDev.size() || names.size() > means.size()) {
        std::cerr << "Error: Not enough standardization parameters for all data entries." << std::endl;
        std::cerr << "Data entries: " << names.size() << ", stdDev size: " << stdDev.size() 
                  << ", means size: " << means.size() << std::endl;
        return 1;
    }
    
    // Apply standardization with division by zero check
    for (size_t i = 0; i < names.size(); i++) {
        // Check for division by zero or very small values
        if (std::abs(stdDev[i]) < 1e-10) {
            std::cerr << "Warning: Near-zero standard deviation for " << names[i] 
                      << " (" << stdDev[i] << "). Setting result to 0." << std::endl;
            scores[i] = 0.0;
        } else {
            // Safe standardization with bounds checking
            try {
                scores[i] = (scores[i] - means[i]) / stdDev[i];
                
                // Check for infinity or NaN
                if (!std::isfinite(scores[i])) {
                    std::cerr << "Warning: Non-finite value produced for " << names[i] 
                              << ". Input: " << scores[i] << ", Mean: " << means[i] 
                              << ", StdDev: " << stdDev[i] << ". Setting to 0." << std::endl;
                    scores[i] = 0.0;
                }
            } catch (const std::exception& e) {
                std::cerr << "Exception during standardization for " << names[i] << ": " 
                          << e.what() << std::endl;
                scores[i] = 0.0;
            }
        }
    }

    return 0;
}


int inference(std::vector<std::string>& input_name, std::vector<double>& input_tensor_values){
    try {
        // Create environment
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXModelInference");
        
        // Session options
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        // Create session
        const char* model_path = "RDP_TripleNN.onnx";
        Ort::Session session(env, model_path, session_options);

        // Get model information
        Ort::AllocatorWithDefaultOptions allocator;    
        
        // Get number of inputs and outputs
        size_t num_input_nodes = session.GetInputCount();
        size_t num_output_nodes = session.GetOutputCount();
        
        // std::cout << "Number of inputs: " << num_input_nodes << std::endl;
        // std::cout << "Number of outputs: " << num_output_nodes << std::endl;
        
        // Get actual input and output names from the model
        std::vector<std::string> input_node_names;
        std::vector<std::string> output_node_names;
        
        // Get input names
        for (size_t i = 0; i < num_input_nodes; i++) {
            auto name = session.GetInputNameAllocated(i, allocator);
            input_node_names.push_back(name.get());
            // std::cout << "Input " << i << " name: " << name.get() << std::endl;
        }
        
        // Get output names
        for (size_t i = 0; i < num_output_nodes; i++) {
            auto name = session.GetOutputNameAllocated(i, allocator);
            output_node_names.push_back(name.get());
            // std::cout << "Output " << i << " name: " << name.get() << std::endl;
        }
        
        // Convert to const char* arrays for the Run() method
        std::vector<const char*> input_names;
        std::vector<const char*> output_names;
        
        for (const auto& name : input_node_names) {
            input_names.push_back(name.c_str());
        }
        
        for (const auto& name : output_node_names) {
            output_names.push_back(name.c_str());
        }
        
        // Get input shape
        Ort::TypeInfo input_type_info = session.GetInputTypeInfo(0);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> input_dims = input_tensor_info.GetShape();
        
        // Print input dimensions
        // std::cout << "Input Dimensions: ";
        // for (auto& dim : input_dims) {
        //     std::cout << dim << " ";
        // }
        // std::cout << std::endl;

        // Create input tensor with correct shape
        std::vector<int64_t> input_shape = {1, static_cast<int64_t>(input_tensor_values.size())};
        
        // Convert double values to float for the tensor
        std::vector<float> float_values(input_tensor_values.begin(), input_tensor_values.end());
          
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info, float_values.data(), float_values.size(),
            input_shape.data(), input_shape.size()));
        
        // Run inference with proper input and output names
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr}, 
            input_names.data(), input_tensors.data(), input_tensors.size(),
            output_names.data(), output_names.size()
        );
        
        // Access the output tensor
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        
        // Get output tensor info
        Ort::TypeInfo output_type_info = session.GetOutputTypeInfo(0);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> output_dims = output_tensor_info.GetShape();
        
        // Print output dimensions
        // std::cout << "Output Dimensions: ";
        // for (auto& dim : output_dims) {
        //     std::cout << dim << " ";
        // }
        // std::cout << std::endl;
        
        // Calculate output size from dimensions (for classification model)
        size_t output_size = 1;
        for (auto& dim : output_dims) {
            if (dim > 0) {
                output_size *= dim;
            }
        }
        
        // Print all classification outputs
        std::cout << "Classification results:" << std::endl;
        for (size_t i = 0; i < output_size; i++) {
            std::cout << "Class " << i << ": " << output_data[i] << std::endl;
        }
        
        // Find the class with the highest probability
        size_t max_index = 0;
        float max_prob = output_data[0];
        for (size_t i = 1; i < output_size; i++) {
            if (output_data[i] > max_prob) {
                max_prob = output_data[i];
                max_index = i;
            }
        }
        std::cout << "Predicted class: " << max_index << " with probability: " << max_prob << std::endl;
        
        return 0;
    }
    catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return 1;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

// Main function
// Reads data from a file, processes it, and prints the results
int main(){
    std::vector<std::string> names;
    std::vector<double> scores;
    

    // Open file for reading
    std::ifstream inFile("sharedData.txt");
    
    if (!inFile.is_open()) {
        std::cerr << "Error: Could not open file for reading." << std::endl;
        return 1;
    }

    std::string line;

    // Read data
    while (std::getline(inFile, line)) {
        std::stringstream ss(line);
        std::string name;
        double score;
        
        // Parse line into name and score
        std::getline(ss, name, ',');
        ss >> score;
        
        names.push_back(name);
        scores.push_back(score);
    }
    
    inFile.close();
    
    //Drop the first element
    int complete = drop(names, scores);

    //Standardise the scores
    complete = standardise(names, scores);

    // Print data to verify
    // for (size_t i = 0; i < names.size(); i++) {
    //     std::cout << names[i] << ": " << scores[i] << std::endl;
    // }

    inference(names, scores);
    
    return 0;
}