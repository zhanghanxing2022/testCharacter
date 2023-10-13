#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

struct X {
    std::vector<std::vector<float>> imageData;
    std::vector<float> labelData;
    std::string filepath;
};

class ImageDataHandler {
public:
    ImageDataHandler(const std::string& dataFolder);
    X getTrain(int i);
    X getTest(int i);

private:
    std::vector<std::vector<X>> data;  // 12x620
    std::vector<int> testIndices;
    std::vector<int> trainIndices;
};

ImageDataHandler::ImageDataHandler(const std::string& dataFolder) {
    // Shuffle indices
    std::random_device rd;
    std::mt19937 g(rd());
    std::vector<int> l(620, 0);
    std::iota(l.begin(), l.end(), 0);

    


    // Initialize indices
    for (int i = 0; i < 12; ++i) {
        std::shuffle(l.begin(), l.end(), g);
        for (int j = 0; j < 620; ++j) {
            int index = i * 620 + l[j];
            if (j < 62) {
                testIndices.push_back(index);
            } else {
                trainIndices.push_back(index);
            }
        }
    }

    std::shuffle(testIndices.begin(), testIndices.end(), g);
    std::shuffle(trainIndices.begin(), trainIndices.end(), g);

    

    // Load data
    data.resize(12);
    for (int i = 1; i <= 12; ++i) {
        for (int j = 1; j <= 620; ++j) {
            std::string filepath = dataFolder + "/" + std::to_string(i) + "/" + std::to_string(j) + ".bmp";

            X x;
            // Simulate loading image data (replace with actual image loading logic)
            // x.imageData = loadImage(filepath);
            x.labelData = std::vector<float>(12, 0.0f);
            x.labelData[i - 1] = 1.0f;  // One-hot encode the label
            x.filepath = filepath;

            data[i - 1].push_back(x);
        }
    }
}

X ImageDataHandler::getTrain(int i) {
    int index = trainIndices[i];
    int category = index / 620;
    int imageIndex = index % 620;

    return data[category][imageIndex];
}

X ImageDataHandler::getTest(int i) {
    int index = testIndices[i];
    int category = index / 620;
    int imageIndex = index % 620;

    return data[category][imageIndex];
}

int main() {
    std::string dataFolder = "path/to/your/data/folder";
    ImageDataHandler handler(dataFolder);

    // Example usage
    for (int i = 0; i < 62; ++i) {
        X trainData = handler.getTrain(i);
        std::cout << "Train Image Path: " << trainData.filepath << std::endl;

        X testData = handler.getTest(i);
        std::cout << "Test Image Path: " << testData.filepath << std::endl;
    }

    return 0;
}
