#include <iostream>
#include <fstream>

int main() {
    const int width = 28;
    const int height = 28;
    const int rowSize = 28; // 每行像素数据的字节数

    std::ifstream bmpFile("../train/1/1.bmp", std::ios::binary);

     if (bmpFile) {
        bmpFile.seekg(58); // 跳过BMP文件头

        unsigned char pixelData[width][height]; // 存储像素数据的数组

        // 读取像素数据
        for (int i = height - 1; i >= 0; --i) {
            bmpFile.read(reinterpret_cast<char*>(&pixelData[i]), rowSize);
            bmpFile.seekg(2, std::ios::cur); // 跳过4字节对齐补充的字节
        }

        // 在这里可以使用读取到的像素数据进行处理
        // 例如，输出像素值
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                std::cout << static_cast<int>(pixelData[i][j]) << " ";
            }
            std::cout << std::endl;
        }

        bmpFile.close();
    } else {
        std::cerr << "Error: Unable to open BMP file." << std::endl;
    }

    return 0;
}











