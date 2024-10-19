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

// CUDA kernel to calculate token occurrences with shared memory
__global__ void calc_token_occurrences_cuda(const char* data, int data_size, const char** words, int* word_lens, int* results, int num_words)
{
    // Shared memory to load chunks of data for the block
    extern __shared__ char shared_data[];

    int tid = threadIdx.x + blockIdx.x * blockDim.x; // Global thread index
    int num_threads = blockDim.x * gridDim.x;        // Total threads across all blocks

    for (int w = 0; w < num_words; ++w) {
        int token_len = word_lens[w];
        const char* token = words[w];

        int numOccurrences = 0;

        for (int i = tid; i < data_size; i += num_threads) {
            // Load data into shared memory
            if (i < data_size) {
                shared_data[threadIdx.x] = data[i];
            }
            __syncthreads(); // Sync to ensure all threads in the block have loaded their data

            if (i < data_size - token_len && compare_token(&shared_data[threadIdx.x], token, token_len)) {
                // Ensure that the prefix and suffix conditions are met
                if ((i == 0 || (data[i - 1] < 'a' || data[i - 1] > 'z')) &&
                    (i + token_len == data_size || (data[i + token_len] < 'a' || data[i + token_len] > 'z'))) {
                    numOccurrences++;
                }
            }

            __syncthreads(); // Sync threads after processing the chunk
        }

        // Store result in global memory
        atomicAdd(&results[w], numOccurrences); // Accumulate results from all threads
    }
}

int main()
{
    const char* filepath = "dataset/shakespeare.txt";
    const int num_iterations = 3; // Number of iterations
    double totalDuration = 0.0; // Total duration across 3 iterations
    std::ofstream iterationData("iterationTimesOp2.csv", std::ofstream::out); // Output file for iteration times

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
    const int num_words = sizeof(words) / sizeof(words[0]);

    // Get the file data size for device allocation
    int data_size = file_data.size();

    // Allocate device memory for the file data
    char* d_data;
    cudaMalloc((void**)&d_data, data_size * sizeof(char));
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
    cudaMemcpy(d_words, d_individual_words, num_words * sizeof(char*), cudaMemcpyHostToDevice);

    // Allocate memory for results on the device
    int* d_results;
    cudaMalloc((void**)&d_results, num_words * sizeof(int));
    cudaMemset(d_results, 0, num_words * sizeof(int)); // Initialize result array to 0

    int threads_per_block = 128; // Block size (128 threads)
    int num_blocks = (data_size + threads_per_block - 1) / threads_per_block; // Number of blocks

    // Loop to run the kernel 3 times and record the duration
    for (int iter = 0; iter < num_iterations; ++iter) {
        auto start = std::chrono::high_resolution_clock::now();

        // Launch the CUDA kernel with optimized grid/block size and shared memory usage
        calc_token_occurrences_cuda << <num_blocks, threads_per_block, threads_per_block * sizeof(char) >> > (d_data, data_size, d_words, d_word_lens, d_results, num_words);

        auto end = std::chrono::high_resolution_clock::now();

        // Copy the results back to the host
        int occurrences[num_words];
        cudaMemcpy(occurrences, d_results, num_words * sizeof(int), cudaMemcpyDeviceToHost);

        // Output the occurrences of each word on the first iteration
        if (iter == 0) {
            for (int i = 0; i < num_words; ++i) {
                std::cout << "Iteration " << iter + 1 << ": Found " << occurrences[i] << " occurrences of word: " << words[i] << std::endl;
            }
        }

        std::chrono::duration<double> duration = end - start;
        totalDuration += duration.count();

        iterationData << "Iteration " << iter + 1 << ", " << duration.count() << " seconds" << std::endl;
    }

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

    iterationData.close();

    return 0;
}
