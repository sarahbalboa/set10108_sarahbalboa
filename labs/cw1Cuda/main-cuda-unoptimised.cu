#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <chrono> // Used for time measurements
#include <cstring>
#include <cuda_runtime.h>

std::vector<char> read_file(const char* filename)
{
    // Open the file in binary mode
    std::ifstream file(filename, std::ios::binary);

    // Check if the file opened successfully
    if (!file) {
        std::cerr << "Error: Could not open the file " << filename << std::endl;
        return {};
    }

    // Move the file cursor to the end of the file to get its size
    file.seekg(0, std::ios::end);
    std::streamsize fileSize = file.tellg();

    // Return the file cursor to the beginning of the file
    file.seekg(0, std::ios::beg);

    // Create a vector of the same size as the file to hold the content
    std::vector<char> buffer(fileSize);

    // Read the entire file into the vector
    if (!file.read(buffer.data(), fileSize)) {
        std::cerr << "Error: Could not read the file content." << std::endl;
        return {};
    }

    // Close the file
    file.close();

    // Output the number of bytes read
    std::cout << "Successfully read " << buffer.size() << " bytes from the file." << std::endl;

    // Convert to lowercase
    std::transform(buffer.begin(), buffer.end(), buffer.begin(), [](char c) { return std::tolower(c); });

    return buffer;
}

// Custom string comparison function for CUDA (works like strncmp)
__device__ bool compare_token(const char* data, const char* token, int token_len)
{
    for (int i = 0; i < token_len; ++i)
    {
        if (data[i] != token[i])
        {
            return false; // If any character doesn't match, return false
        }
    }
    return true;
}

// CUDA kernel to calculate token occurrences
__global__ void calc_token_occurrences_cuda(const char* data, int data_size, const char* token, int token_len, int* result)
{
    int numOccurrences = 0;

    // Single thread for simplicity
    for (int i = 0; i < data_size; ++i)
    {
        // test 1: does this match the token?
        if (!compare_token(&data[i], token, token_len))
            continue;

        // test 2: is the prefix a non-letter character?
        auto iPrefix = i - 1;
        if (iPrefix >= 0 && data[iPrefix] >= 'a' && data[iPrefix] <= 'z')
            continue;

        // test 3: is the suffix a non-letter character?
        auto iSuffix = i + token_len;
        if (iSuffix < data_size && data[iSuffix] >= 'a' && data[iSuffix] <= 'z')
            continue;

        // Increment occurrence count
        numOccurrences++;
    }

    // Write result back to host
    *result = numOccurrences;
}

int main()
{
    // Example chosen file
    const char* filepath = "dataset/shakespeare.txt";
    const int num_iterations = 3; // Number of iterations
    double totalDuration = 0.0;
    std::ofstream iterationData("iterationTimesUnop.csv", std::ofstream::out); // Open CSV for iteration times

    if (!iterationData) {
        std::cerr << "Error: Could not open iterationTimes.csv" << std::endl;
        return -1;
    }

     
    std::vector<char> file_data = read_file(filepath);
    if (file_data.empty())
        return -1;
    // Example word list
    const char* words[] = { "sword", "fire", "death", "love", "hate", "the", "man", "woman" };
    
    //get the file data size for device allocation
    int data_size = file_data.size();

    // Start total timer
    auto totalStart = std::chrono::high_resolution_clock::now();

    // Allocate device memory for the file data
    char* d_data;
    cudaMalloc((void**)&d_data, data_size * sizeof(char));
    //copy the file data from host memory to device memory 
    cudaMemcpy(d_data, file_data.data(), data_size * sizeof(char), cudaMemcpyHostToDevice);

    

    // Iterate num_iterations times
    for (int iter = 0; iter < num_iterations; ++iter) {
        double iterationDuration = 0.0; // Time for this iteration

        // Loop through each word
        for (const char* word : words) {
            int token_len = strlen(word);
            int* d_occurrences;
            int occurrences;

            // Allocate memory for the result on the device
            cudaMalloc((void**)&d_occurrences, sizeof(int));

            // Copy the token to the device
            char* d_token;
            cudaMalloc((void**)&d_token, token_len * sizeof(char));
            cudaMemcpy(d_token, word, token_len * sizeof(char), cudaMemcpyHostToDevice);

            // Start timer for word
            auto start = std::chrono::high_resolution_clock::now();

            // Launch CUDA kernel (1 block and 1 thread per word)
            calc_token_occurrences_cuda << <1, 1 >> > (d_data, data_size, d_token, token_len, d_occurrences);

            // End timer for word
            auto end = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> duration = end - start;
            iterationDuration += duration.count(); // Add to iteration time

            // Copy the result back to the host
            cudaMemcpy(&occurrences, d_occurrences, sizeof(int), cudaMemcpyDeviceToHost);

            if (iter == 0) {
                std::cout << "Iteration " << iter + 1 << " - Found " << occurrences << " occurrences of word: " << word << std::endl;

            }

            // Free device memory for the current word
            cudaFree(d_token);
            cudaFree(d_occurrences);
        }

        // Record the time for this iteration in the CSV file
        iterationData << "Iteration " << iter + 1 << ", " << iterationDuration << " seconds" << std::endl;

        // Accumulate the total duration for averaging later
        totalDuration += iterationDuration;
    }

    // Calculate the average execution time
    double averageDuration = totalDuration / num_iterations;

    // Print average duration
    std::cout << "Average CUDA execution time over " << num_iterations << " iterations: " << averageDuration << " seconds" << std::endl;

    // Free device memory
    cudaFree(d_data);

    iterationData.close(); // Close the CSV file

    return 0;
}
