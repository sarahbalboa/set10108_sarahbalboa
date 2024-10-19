#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <chrono> // Used for time measurements
#include <cstring>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

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

// CUDA kernel to calculate token occurrences, each thread handles one word
__global__ void calc_token_occurrences_cuda(const char* data, int data_size, const char** words, int* word_lens, int* results, int num_words)
{
    int idx = threadIdx.x; // Each thread handles one word

    if (idx < num_words) {
        const char* token = words[idx];     // Each thread gets its own word
        int token_len = word_lens[idx];     // Get the length of the word
        int numOccurrences = 0;

        // Loop through the data to search for the word
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

        // Write result back to the results array
        results[idx] = numOccurrences;
    }
}

int main()
{
    // Example chosen file
    const char* filepath = "dataset/shakespeare.txt";
    const int num_iterations = 3; // Number of iterations
    double totalDuration = 0.0; // Total duration across 3 iterations
    std::ofstream iterationData("iterationTimesOp1.csv", std::ofstream::out); // Output file for iteration times
    
    if (!iterationData) {
        std::cerr << "Error: Could not open iterationTimes.csv" << std::endl;
        return -1;
    }

    // Read the file into a vector
    std::vector<char> file_data = read_file(filepath);
    if (file_data.empty())
        return -1;

    // Example word list
    const char* words[] = { "sword", "fire", "death", "love", "hate", "the", "man", "woman" };
    const int num_words = sizeof(words) / sizeof(words[0]); // number of words constant used for number of threads required
    
    // Get the file data size for device allocation
    int data_size = file_data.size(); 

    // Allocate device memory for the file data
    char* d_data;
    cudaMalloc((void**)&d_data, data_size * sizeof(char));
    // Copy the file data from host memory to device memory
    cudaMemcpy(d_data, file_data.data(), data_size * sizeof(char), cudaMemcpyHostToDevice);


    // Allocate device memory for the words and word lengths
    const char** d_words;
    int* d_word_lens;
    cudaMalloc((void**)&d_words, num_words * sizeof(char*));
    cudaMalloc((void**)&d_word_lens, num_words * sizeof(int));

    // Prepare word lengths and copy words and lengths to the device
    int word_lens[num_words];
    for (int i = 0; i < num_words; ++i) {
        word_lens[i] = strlen(words[i]);
    }
    cudaMemcpy(d_word_lens, word_lens, num_words * sizeof(int), cudaMemcpyHostToDevice);

    // Copy words individually to the device
    char* d_individual_words[num_words];  // This will hold the device pointers for each word
    for (int i = 0; i < num_words; ++i) {
        cudaMalloc((void**)&d_individual_words[i], word_lens[i] * sizeof(char));
        cudaMemcpy(d_individual_words[i], words[i], word_lens[i] * sizeof(char), cudaMemcpyHostToDevice);
    }
    // Copy the array of word pointers to the device
    cudaMemcpy(d_words, d_individual_words, num_words * sizeof(char*), cudaMemcpyHostToDevice);

    // Allocate memory for results on the device
    int* d_results;
    cudaMalloc((void**)&d_results, num_words * sizeof(int));

    // Loop to run the kernel 3 times and record the duration
    for (int iter = 0; iter < num_iterations; ++iter) {
        // Start timer ---------------------------------------------------------------------------------
        auto start = std::chrono::high_resolution_clock::now();

        // Launch the CUDA kernel with num_words threads (one per word)
        calc_token_occurrences_cuda << <1, num_words >> > (d_data, data_size, d_words, d_word_lens, d_results, num_words);

        // End timer ---------------------------------------------------------------------------------
        auto end = std::chrono::high_resolution_clock::now();

        // Copy the results back to the host
        int occurrences[num_words];
        cudaMemcpy(occurrences, d_results, num_words * sizeof(int), cudaMemcpyDeviceToHost);

        // Output the occurrences of each word only on the firts iteration
        if (iter == 0) {
            for (int i = 0; i < num_words; ++i) {
                std::cout << "Iteration " << iter + 1 << ": Found " << occurrences[i] << " occurrences of word: " << words[i] << std::endl;
            }
        }

        // Calculate total duration for this iteration
        std::chrono::duration<double> duration = end - start;
        totalDuration += duration.count();

        // Write duration for this iteration to CSV
        iterationData << "Iteration " << iter + 1 << ", " << duration.count() << " seconds" << std::endl;
    }

    // Print total average duration
    double averageDuration = totalDuration / 3.0;
    std::cout << "Average CUDA execution time for 3 iterations: " << averageDuration << " seconds" << std::endl;

    // Free device memory
    cudaFree(d_data);
    cudaFree(d_words);
    cudaFree(d_word_lens);
    cudaFree(d_results);

    // Free the device memory for individual words
    for (int i = 0; i < num_words; ++i) {
        cudaFree(d_individual_words[i]);
    }

    // Close the CSV file
    iterationData.close();

    return 0;
}
