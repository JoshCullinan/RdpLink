#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cstdlib>
#include <memory>
#include <array>

// Include PSNN header to use its declarations
#include "PSNN.h"

using namespace std;

// Class names for prediction results
const std::array<std::string, 3> CLASS_NAMES = {
    "Class 0", 
    "Class 1", 
    "Class 2"
};

/**
 * Execute PSNN program and capture its output
 * 
 * @return exit code of the PSNN program
 */
int runPSNNAndCaptureOutput(std::string& output) {
    // Create a pipe to capture output
    FILE* pipe = popen("./PSNN", "r");
    if (!pipe) {
        cerr << "Error: Failed to run PSNN program" << endl;
        return -1;
    }
    
    // Read output from pipe
    std::array<char, 256> buffer;
    output.clear();
    
    while (!feof(pipe)) {
        if (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
            output += buffer.data();
        }
    }
    
    // Close pipe and get exit code
    int exitStatus = pclose(pipe);
    int exitCode = WEXITSTATUS(exitStatus);
    
    if (WIFEXITED(exitStatus)) {
        cout << "PSNN exited with status: " << exitCode << endl;
    } else if (WIFSIGNALED(exitStatus)) {
        cerr << "PSNN was terminated by signal: " << WTERMSIG(exitStatus) << endl;
        return -1;
    }
    
    return exitCode;
}

/**
 * Parse prediction results from PSNN output
 * 
 * @param output The captured output from PSNN
 * @param probabilities Vector to store class probabilities
 * @return predicted class index
 */
int parsePredictionResults(const std::string& output, std::vector<float>& probabilities) {
    probabilities.clear();
    probabilities.resize(3, 0.0f);
    
    int predictedClass = -1;
    
    // Print the output for debugging
    cout << "\n--- PSNN Output (Debug) ---" << endl;
    cout << output << endl;
    cout << "--- End of PSNN Output ---\n" << endl;
    
    // Parse the output to find class probabilities and predicted class
    size_t classPos = output.find("Classification results:");
    if (classPos != std::string::npos) {
        try {
            // Parse class probabilities
            for (int i = 0; i < 3; i++) {
                std::string searchStr = "Class " + std::to_string(i) + ": ";
                size_t pos = output.find(searchStr, classPos);
                if (pos != std::string::npos) {
                    pos += searchStr.length();
                    size_t endPos = output.find('\n', pos);
                    if (endPos != std::string::npos) {
                        std::string probStr = output.substr(pos, endPos - pos);
                        try {
                            probabilities[i] = std::stof(probStr);
                        } catch (const std::exception& e) {
                            cerr << "Error parsing probability for Class " << i << ": " << e.what() << endl;
                        }
                    }
                }
            }
            
            // Find the predicted class from the output
            size_t predPos = output.find("Predicted class:");
            if (predPos != std::string::npos) {
                // Try to extract the class number, this is a bit tricky with the formatting
                std::string line = output.substr(predPos);
                std::string search = "Predicted class: ";
                size_t pos = line.find(search);
                if (pos != std::string::npos) {
                    try {
                        // Extract just the digit
                        char digit = line[pos + search.length()];
                        if (isdigit(digit)) {
                            predictedClass = digit - '0';
                            cout << "Successfully parsed predicted class: " << predictedClass << endl;
                        } else {
                            // Failed to parse, find max probability
                            predictedClass = 0;
                            float maxProb = probabilities[0];
                            for (size_t i = 1; i < probabilities.size(); i++) {
                                if (probabilities[i] > maxProb) {
                                    maxProb = probabilities[i];
                                    predictedClass = i;
                                }
                            }
                            cout << "Using max probability class instead: " << predictedClass << endl;
                        }
                    } catch (const std::exception& e) {
                        cerr << "Error parsing predicted class: " << e.what() << endl;
                        predictedClass = 0; // Default to first class
                    }
                }
            }
        } catch (const std::exception& e) {
            cerr << "Error during parsing: " << e.what() << endl;
        }
    } else {
        // No classification results found, determine max probability manually
        cout << "Classification results not found in output, using highest probability class" << endl;
        
        // If we couldn't find the class in the output, find the max manually
        float maxProb = probabilities[0];
        predictedClass = 0;
        
        for (size_t i = 1; i < probabilities.size(); i++) {
            if (probabilities[i] > maxProb) {
                maxProb = probabilities[i];
                predictedClass = i;
            }
        }
    }
    
    return predictedClass;
}

int main() {
    //Vector containing input statistics
    // Names of inputs:
    vector<string> names{"ListCorr(A)1","ListCorr(A)2","ListCorr(A)3","SimScoreB(A)1","SimScoreB(A)2","SimScoreB(A)3","SimScore(A)1","SimScore(A)2","SimScore(A)3",
        "PhPrScore(A)1","PhPrScore(A)2","PhPrScore(A)3","PhPrScore2(A)1","PhPrScore2(A)2","PhPrScore2(A)3","PhPrScore3(A)1","PhPrScore3(A)2","PhPrScore3(A)3",
        "SubScore(A)1","SubScore(A)2","SubScore(A)3","SSDist(A)1","SSDist(A)2","SSDist(A)3","OUIndexA(A)1","OUIndexA(A)2","OUIndexA(A)3","SubPhPrScore(A)1","SubPhPrScore(A)2","SubPhPrScore(A)3",
        "SubScore2(A)1","SubScore2(A)2","SubScore2(A)3","SubPhPrScore2(A)1","SubPhPrScore2(A)2","SubPhPrScore2(A)3","SRCompatF(A)1","SRCompatF(A)2","SRCompatF(A)3","SRCompatS(A)1","SRCompatS(A)2","SRCompatS(A)3",
        "RCompat(A)1","RCompat(A)2","RCompat(A)3","RCompat2(A)1", "RCompat2(A)2", "RCompat2(A)3", "RCompat3(A)1", "RCompat3(A)2", "RCompat3(A)3", "RCompat4(A)1", "RCompat4(A)2", "RCompat4(A)3", 
        "RCompatS(A)1", "RCompatS(A)2", "RCompatS(A)3", "RCompatS2(A)1", "RCompatS2(A)2", "RCompatS2(A)3", "RCompatS3(A)1", "RCompatS3(A)2", "RCompatS3(A)3", "RCompatS4(A)1", "RCompatS4(A)2", "RCompatS4(A)3", 
        "RCompatXF(A)1", "RCompatXF(A)2", "RCompatXF(A)3","RCompatXS(A)1", "RCompatXS(A)2", "RCompatXS(A)3", "RCompatC(A)1", "RCompatC(A)2", "RCompatC(A)3", "RCompatD(A)1", "RCompatD(A)2", "RCompatD(A)3",
        "TrpScore(A)1", "TrpScore(A)2", "TrpScore(A)3", "BadDists(A)1", "BadDists(A)2", "BadDists(A)3", "OUList(A)1", "OUList(A)2", "OUList(A)3", "ListCorr2(A)1", "ListCorr2(A)2", "ListCorr2(A)3",
        "ListCorr3(A)1", "ListCorr3(A)2", "ListCorr3(A)3", "Consensus(A:0)1", "Consensus(A:0)2", "Consensus(A:0)3", "Consensus(A:1)1", "Consensus(A:1)2", "Consensus(A:1)3",
        "Consensus(A:2)1", "Consensus(A:2)2", "Consensus(A:2)3", "OuCheck(A)1", "OuCheck(A)2", "OuCheck(A)3", "SetTot(0:A)1", "SetTot(0:A)2", "SetTot(0:A)3", "SetTot(1:A)1", "SetTot(1:A)2", "SetTot(1:A)3",
        "RankF(A:0)1", "RankF(A:0)2", "RankF(A:0)3", "RankF(A:1)1", "RankF(A:1)2", "RankF(A:1)3", "dMax(A)1", "dMax(A)2", "dMax(A)3"
    };

    vector<float> input {19.0,30.0,20.0,0.22937,0.01384,-0.01384,1.0,0.0,0.0,0.65521,0.48745,0.78721,0.97112,0.15106,0.86357,0.97112,0.15106,0.86357,3.447357,1.946852,2.164306,0.03027,0.03633,0.03058,0.0,1.0,0.0,0.6365,0.7738,0.5749,0.183,0.291,0.221,0.51605,0.92983,0.56319,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,3.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,7.219661,5.75,2.061862,1.0,0.0,3.0,1.0,0.0,2.0,0.74117,0.17592,0.52591,0.10402,0.10604,0.10739,5.0,8.0,-1.0,9.0,9.0,6.0,89.73364,175.2845,44.48182,1.0,-1.0,-1.0,15.0,10.0,15.0,0.0,0.0,0.0,85.0,75.0,44.0,0.0,85.0,2.0,0.4513639,0.4882911,0.4009681};

    cout << "=== PSNN Tester ===" << endl;
    cout << "Preparing input data..." << endl;
    
    // Open file to write to
    std::ofstream outFile("sharedData.txt");
    if (!outFile.is_open()) {
        cerr << "Error: Could not open sharedData.txt for writing." << endl;
        return 1;
    }

    // Write data to file
    for (size_t i = 0; i < names.size(); i++) {
        outFile << names[i] << "," << input[i] << endl;
    }

    // Close file
    outFile.close();
    cout << "Data written to sharedData.txt" << endl;
    
    // Run PSNN and capture output
    std::string psnnOutput;
    cout << "Running PSNN for prediction..." << endl;
    int exitCode = runPSNNAndCaptureOutput(psnnOutput);
    
    if (exitCode != 0) {
        cerr << "Error: PSNN program exited with code " << exitCode << endl;
        return exitCode;
    }
    
    // Parse results
    std::vector<float> probabilities;
    int predictedClass = parsePredictionResults(psnnOutput, probabilities);
    
    // Ensure we have a valid class index (default to 0 if not)
    if (predictedClass < 0 || predictedClass >= 3) {
        cout << "Invalid predicted class, defaulting to class with highest probability" << endl;
        
        // Find the class with highest probability
        predictedClass = 0;
        float maxProb = probabilities[0];
        for (size_t i = 1; i < probabilities.size(); i++) {
            if (probabilities[i] > maxProb) {
                maxProb = probabilities[i];
                predictedClass = i;
            }
        }
    }
    
    // Display prediction results
    cout << "\n=== Prediction Results ===" << endl;
    for (size_t i = 0; i < probabilities.size(); i++) {
        cout << CLASS_NAMES[i] << ": " << probabilities[i] * 100.0f << "%" << endl;
    }
    
    cout << "\nPredicted: " << CLASS_NAMES[predictedClass] 
         << " (" << probabilities[predictedClass] * 100.0f << "%)" << endl;
         
    // Write results to file
    std::ofstream resultFile("prediction_result.txt");
    if (resultFile.is_open()) {
        // Write class names header
        for (const auto& className : CLASS_NAMES) {
            resultFile << className << ",";
        }
        resultFile << endl;
        
        // Write probabilities
        for (const auto& prob : probabilities) {
            resultFile << prob << ",";
        }
        resultFile << endl;
        
        // Write the final prediction
        resultFile << "Predicted: " << CLASS_NAMES[predictedClass] 
                  << " with " << probabilities[predictedClass] * 100.0f << "% confidence" << endl;
        
        resultFile.close();
        cout << "Results saved to prediction_result.txt" << endl;
    } else {
        cerr << "Warning: Could not open prediction_result.txt for writing" << endl;
    }
    
    return 0;
}



     



