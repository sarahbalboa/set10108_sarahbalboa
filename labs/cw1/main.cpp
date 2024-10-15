#include <filesystem>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <chrono> // Used for time measurements

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

int calc_token_occurrences(const std::vector<char>& data, const char* token)
{
    int numOccurrences = 0;
    int tokenLen = int(strlen(token));
    for (int i = 0; i< int(data.size()); ++i)
    {
        // test 1: does this match the token?
        auto diff = strncmp(&data[i], token, tokenLen);
        if (diff != 0)
            continue;

        // test 2: is the prefix a non-letter character?
        auto iPrefix = i - 1;
        if (iPrefix >= 0 && data[iPrefix] >= 'a' && data[iPrefix] <= 'z')
            continue;

        // test 3: is the prefix a non-letter character?
        auto iSuffix = i + tokenLen;
        if (iSuffix < int(data.size()) && data[iSuffix] >= 'a' && data[iSuffix] <= 'z')
            continue;
        ++numOccurrences;
    }
    return numOccurrences;
}

int main()
{
    // Example chosen file
    const char * filepath = "dataset/shakespeare.txt";
    const int numRuns = 10;
    double totalDuration = 0.0;


    std::vector<char> file_data = read_file(filepath);
    if (file_data.empty())
        return -1;

    // Start total timer
    auto totalStart = std::chrono::high_resolution_clock::now();

    // Example word list
    const char * words[] = {"sword", "fire", "death", "love", "hate", "the", "man", "woman"};
    for (int i = 0; i < numRuns; ++i) {

        // Timing the actual search function
        auto start = std::chrono::high_resolution_clock::now();

        //run funtion
        for (const char* word : words)
        {
            int occurrences = calc_token_occurrences(file_data, word);
            if(i == 0) //print output of occurrences only once
            {
                std::cout << "Found " << occurrences << " occurrences of word: " << word << std::endl;
            }
        }

        //end timer
        auto end = std::chrono::high_resolution_clock::now();

        //calculate duration
        std::chrono::duration<double> duration = end - start;
        totalDuration += duration.count();

        //print run duration
        std::cout << "Run Number: " << i + 1 <<" - Time: " << duration.count() << " seconds" << std::endl;
    }

    double averageDuration = totalDuration / numRuns;

    //print average duration
    std::cout << "Average CUDA execution time (" << numRuns <<" times run): " << averageDuration << " seconds" << std::endl;

    return 0;
}