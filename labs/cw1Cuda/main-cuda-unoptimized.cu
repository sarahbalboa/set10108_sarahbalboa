//single block thread simple
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <chrono>
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

    // convert to lowercase
    std::transform(buffer.begin(), buffer.end(), buffer.begin(), [](char c) { return std::tolower(c); });

    return buffer;
}

// CUDA version of calc_token_occurrences
__global__ void calc_token_occurrences_CUDA(const char* data, int data_size, const char* token, int token_length, int* numOccurrences)
{
    int count = 0;

    for (int i = 0; i < data_size; ++i)
    {
        // test 1: does this match the token?
        bool match = true;
        for (int j = 0; j < token_length; ++j)
        {
            if (i + j >= data_size || data[i + j] != token[j])
            {
                match = false;
                break;
            }
        }

        // test 2: is the prefix a non-letter character?
        if (match)
        {
            bool validPrefix = (i == 0) || (data[i - 1] < 'a' || data[i - 1] > 'z');
            
            if (validPrefix)
            {
                count++;
            }
        }

        // test 3: is the sufix a non-letter character?
        if (match)
        {
            
            bool validSuffix = (i + token_length == data_size) || (data[i + token_length] < 'a' || data[i + token_length] > 'z');

            if (validSuffix)
            {
                count++;
            }
        }
    }

    *numOccurrences = count;
}

int main()
{
    // Example chosen file
    const char* filepath = "dataset/shakespeare.txt";
    std::ofstream data("dataCudaUnop.csv", std::ofstream::out);
    const int numRuns = 50;
    double totalDuration = 0.0;

    std::vector<char> file_data = read_file(filepath);
    if (file_data.empty())
        return -1;

    // Start total timer
    auto totalStart = std::chrono::high_resolution_clock::now();

    // Example word list
    const char* words[] = { "sword", "fire", "death", "love", "hate", "the", "man", "woman" };
    int numWords = sizeof(words) / sizeof(words[0]);

    // Transfer file data to device memory
    char* d_data;
    size_t data_size = file_data.size();
    cudaMalloc((void**)&d_data, data_size);
    cudaMemcpy(d_data, file_data.data(), data_size, cudaMemcpyHostToDevice);

    // Allocate memory for occurrences result on device
    int* d_numOccurrences;
    cudaMalloc((void**)&d_numOccurrences, sizeof(int));

    for (int i = 0; i < numRuns; ++i) {
        // Timing the actual search function
        auto start = std::chrono::high_resolution_clock::now();

        // Run token search for each word
        for (const char* word : words) {
            int token_length = strlen(word);

            // Call the CUDA kernel with 1 block and 1 thread
            calc_token_occurrences_CUDA <<<1, 1 >>>(d_data, data_size, word, token_length, d_numOccurrences);

            // Wait for GPU to finish execution
            cudaDeviceSynchronize();

            // Get result back from device
            int numOccurrences;
            cudaMemcpy(&numOccurrences, d_numOccurrences, sizeof(int), cudaMemcpyDeviceToHost);

            if (i == 0) { // Print result only for the first run
                std::cout << "Found " << numOccurrences << " occurrences of word: " << word << std::endl;
            }
        }

        // End timer
        auto end = std::chrono::high_resolution_clock::now();

        // Calculate duration
        std::chrono::duration<double> duration = end - start;
        totalDuration += duration.count();
        data << duration.count() << std::endl;
    }

    double averageDuration = totalDuration / numRuns;

    // Print average duration
    std::cout << "Average CUDA execution time (" << numRuns << " times run): " << averageDuration << " seconds" << std::endl;

    // Free device memory
    cudaFree(d_data);
    cudaFree(d_numOccurrences);
    data.close();

    return 0;
}
