// PSNN.cpp - Windows DLL version
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <onnxruntime_cxx_api.h>

// Windows-specific DLL export macros
#ifdef _WIN32
    #ifdef PSNN_EXPORTS
        #define PSNN_API __declspec(dllexport)
    #else
        #define PSNN_API __declspec(dllimport)
    #endif
#else
    #define PSNN_API
#endif

// Structure for returning predictions
struct PredictionResult {
    float class_probabilities[3];  // Fixed size for 3 classes
    int predicted_class;
    float confidence;
};

// Static vectors to hold standardisation parameters
static const std::vector<std::string> DROP_NAMES{
    "SRCompatF(A)1","SRCompatF(A)2","SRCompatF(A)3",
    "SRCompatS(A)1","SRCompatS(A)2", "SRCompatS(A)3",
    "RCompatXF(A)1","RCompatXF(A)2","RCompatXF(A)3",
    "RCompatXS(A)1", "RCompatXS(A)2","RCompatXS(A)3",
    "SetTot(1:A)1","SetTot(1:A)2","SetTot(1:A)3", 
    "Consensus(A:0)1", "Consensus(A:0)2","Consensus(A:0)3", 
    "Consensus(A:1)1", "Consensus(A:1)2", "Consensus(A:1)3",
    "Consensus(A:2)1", "Consensus(A:2)2", "Consensus(A:2)3"
};

static const std::vector<double> STD_DEV{
    77.81935754, 78.61500891, 78.49932003, 0.33135391, 0.3302249,
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
    47.81972043, 47.4848627, 47.60851212,  0.21678955,  0.21445225,0.21430079
};

static const std::vector<double> MEANS{
    93.64829657377209,94.9787838666968,94.9850292246585,0.1652282657021991,0.13167896018342107,
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
    60.04446023185972,58.992721283947425,58.86758161914296,0.528549646892492,0.5111502837005976,0.5075989531249846
};

// Internal class to handle ONNX session
class ONNXInference {
private:
    Ort::Env env;
    Ort::Session* session;
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    
public:
    ONNXInference(const char* model_path) : env(ORT_LOGGING_LEVEL_WARNING, "ONNXModelInference") {
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        session = new Ort::Session(env, model_path, session_options);
        
        // Get input and output names
        size_t num_input_nodes = session->GetInputCount();
        size_t num_output_nodes = session->GetOutputCount();
        
        for (size_t i = 0; i < num_input_nodes; i++) {
            auto name = session->GetInputNameAllocated(i, allocator);
            input_names.push_back(_strdup(name.get()));
        }
        
        for (size_t i = 0; i < num_output_nodes; i++) {
            auto name = session->GetOutputNameAllocated(i, allocator);
            output_names.push_back(_strdup(name.get()));
        }
    }
    
    ~ONNXInference() {
        delete session;
        for (auto& name : input_names) free((void*)name);
        for (auto& name : output_names) free((void*)name);
    }
    
    bool runInference(const std::vector<float>& input_values, std::vector<float>& output_probs) {
        try {
            // Create input tensor
            std::vector<int64_t> input_shape = {1, static_cast<int64_t>(input_values.size())};
            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            
            std::vector<Ort::Value> input_tensors;
            input_tensors.push_back(Ort::Value::CreateTensor<float>(
                memory_info, const_cast<float*>(input_values.data()), input_values.size(),
                input_shape.data(), input_shape.size()));
            
            // Run inference
            auto output_tensors = session->Run(
                Ort::RunOptions{nullptr}, 
                input_names.data(), input_tensors.data(), input_tensors.size(),
                output_names.data(), output_names.size()
            );
            
            // Get output data
            float* output_data = output_tensors[0].GetTensorMutableData<float>();
            
            // Get output size
            auto output_tensor_info = output_tensors[0].GetTensorTypeAndShapeInfo();
            std::vector<int64_t> output_dims = output_tensor_info.GetShape();
            
            size_t output_size = 1;
            for (auto& dim : output_dims) {
                if (dim > 0) {
                    output_size *= dim;
                }
            }
            
            // Copy results
            output_probs.clear();
            output_probs.assign(output_data, output_data + output_size);
            
            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "Inference error: " << e.what() << std::endl;
            return false;
        }
    }
};

// Global instance (initialized on first use)
static ONNXInference* g_inference = nullptr;

// DLL entry point
#ifdef _WIN32
#include <windows.h>

BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved) {
    switch (ul_reason_for_call) {
        case DLL_PROCESS_ATTACH:
            // Initialize when DLL is loaded
            break;
        case DLL_PROCESS_DETACH:
            // Cleanup when DLL is unloaded
            if (g_inference) {
                delete g_inference;
                g_inference = nullptr;
            }
            break;
    }
    return TRUE;
}
#endif

// The main function that RDP will call
extern "C" {

/**
 * Initialize the PSNN model
 * 
 * @param model_path Path to the ONNX model file
 * @return true if successful, false otherwise
 */
PSNN_API bool PSNN_Initialize(const char* model_path) {
    try {
        if (g_inference) {
            delete g_inference;
        }
        g_inference = new ONNXInference(model_path);
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Initialization error: " << e.what() << std::endl;
        return false;
    }
}

/**
 * Process input data and return predictions
 * 
 * @param names Array of feature names
 * @param values Array of feature values
 * @param num_features Number of features in the arrays
 * @param result Output structure for predictions
 * @return true if successful, false otherwise
 */
PSNN_API bool PSNN_Predict(const char** names, const double* values, int num_features, PredictionResult* result) {
    if (!g_inference || !names || !values || !result) {
        return false;
    }
    
    // Copy data into vectors for processing
    std::vector<std::string> feature_names;
    std::vector<double> feature_values;
    
    for (int i = 0; i < num_features; i++) {
        feature_names.push_back(names[i]);
        feature_values.push_back(values[i]);
    }
    
    // Drop features
    for (size_t i = 0; i < feature_names.size(); ) {
        if (std::find(DROP_NAMES.begin(), DROP_NAMES.end(), feature_names[i]) != DROP_NAMES.end()) {
            feature_names.erase(feature_names.begin() + i);
            feature_values.erase(feature_values.begin() + i);
        } else {
            ++i;
        }
    }
    
    // Standardize
    if (feature_names.size() > STD_DEV.size() || feature_names.size() > MEANS.size()) {
        return false;
    }
    
    for (size_t i = 0; i < feature_names.size(); i++) {
        if (std::abs(STD_DEV[i]) < 1e-10) {
            feature_values[i] = 0.0;
        } else {
            feature_values[i] = (feature_values[i] - MEANS[i]) / STD_DEV[i];
        }
    }
    
    // Convert to float for inference
    std::vector<float> float_values(feature_values.begin(), feature_values.end());
    
    // Run inference
    std::vector<float> output_probs;
    if (!g_inference->runInference(float_values, output_probs)) {
        return false;
    }
    
    // Fill result structure
    for (size_t i = 0; i < 3 && i < output_probs.size(); i++) {
        result->class_probabilities[i] = output_probs[i];
    }
    
    // Find predicted class
    int max_index = 0;
    float max_prob = output_probs[0];
    for (size_t i = 1; i < output_probs.size(); i++) {
        if (output_probs[i] > max_prob) {
            max_prob = output_probs[i];
            max_index = i;
        }
    }
    
    result->predicted_class = max_index;
    result->confidence = max_prob;
    
    return true;
}

/**
 * Cleanup resources
 */
PSNN_API void PSNN_Cleanup() {
    if (g_inference) {
        delete g_inference;
        g_inference = nullptr;
    }
}

} // extern "C"