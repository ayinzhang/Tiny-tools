#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <map>
#include <cmath>
using namespace cv;
using namespace std;
namespace fs = filesystem;

class ImageProcessor 
{
    vector<string> imageFiles;
    size_t currentIndex = 0;
    Mat currentImage;
    Mat displayImage;
    Mat originalImage;
    Rect cropRect;
    Point dragStart;
    bool isDragging = false;
    int targetWidth = 512;
    int targetHeight = 512;
    bool batchMode = false;
    double scale = 1.0;

    const vector<string> supportedFormats = { ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp" };

public:
    ImageProcessor() 
    {
        scanCurrentDirectory();
        analyzeDimensions();
    }

    void scanCurrentDirectory() 
    {
        cout << "扫描当前文件夹中的图片文件..." << endl;

        for (const auto& entry : fs::directory_iterator(".")) 
        {
            if (entry.is_regular_file()) 
            {
                string extension = entry.path().extension().string();
                transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

                if (find(supportedFormats.begin(), supportedFormats.end(), extension) != supportedFormats.end()) 
                    imageFiles.push_back(entry.path().string());
            }
        }

        sort(imageFiles.begin(), imageFiles.end());
        cout << "找到 " << imageFiles.size() << " 个图片文件" << endl;
    }

    void analyzeDimensions() 
    {
        if (imageFiles.empty()) return;

        map<pair<int, int>, int> dimensionCount;
        vector<int> widths, heights;

        for (const auto& file : imageFiles) 
        {
            Mat img = imread(file, IMREAD_UNCHANGED);
            if (!img.empty()) 
            {
                dimensionCount[{img.cols, img.rows}]++;
                widths.push_back(img.cols); heights.push_back(img.rows);
            }
        }

        if (widths.empty()) return;

        auto mostCommon = max_element(
            dimensionCount.begin(), dimensionCount.end(), [](const auto& a, const auto& b) { return a.second < b.second; });

        if (mostCommon != dimensionCount.end()) 
        {
            int commonWidth = mostCommon->first.first;
            int commonHeight = mostCommon->first.second;

            // 计算建议的64倍数尺寸
            targetWidth = 64 * static_cast<int>(round(static_cast<float>(commonWidth) / 64));
            targetHeight = 64 * static_cast<int>(round(static_cast<float>(commonHeight) / 64));

            cout << "尺寸分析结果:" << endl;
            cout << "最常见尺寸: " << commonWidth << "x" << commonHeight << endl;
            cout << "建议尺寸: " << targetWidth << "x" << targetHeight << endl;
        }
    }

    void selectProcessingMode() 
    {
        if (imageFiles.empty()) return;

        cout << "\n=== 选择处理模式 ===" << endl;
        cout << "1. 批量处理 - 自动居中裁剪所有图片" << endl;
        cout << "2. 逐张处理 - 手动拖动选择框定位" << endl;
        cout << "请选择模式 (1 或 2): ";

        int choice; cin >> choice;

        if (choice == 1) { batchMode = true; batchProcess(); }
        else if(choice == 2) { batchMode = false; setupSingleImageMode(); }
    }

    void setupSingleImageMode() 
    {
        if (imageFiles.empty()) return;

        // 初始化裁剪框（居中位置）
        loadCurrentImage();
        initializeCropRect();

        namedWindow("LoRA图片预处理工具 - 逐张处理模式");
        setMouseCallback("LoRA图片预处理工具 - 逐张处理模式",
            [](int event, int x, int y, int flags, void* userdata) 
            {static_cast<ImageProcessor*>(userdata)->mouseCallback(event, x, y, flags);}, this);

        cout << "\n=== 逐张处理模式 ===" << endl;
        cout << "目标尺寸: " << targetWidth << "x" << targetHeight << endl;
        cout << "操作说明:" << endl;
        cout << "鼠标拖拽 - 移动选择框（框大小固定）" << endl;
        cout << "鼠标滚轮 - 放大/缩小图片（框大小不变）" << endl;
        cout << "S - 保存并替换原图" << endl;
        cout << "D - 删除当前图片" << endl;
        cout << "N - 下一张图片" << endl;
        cout << "P - 上一张图片" << endl;
        cout << "R - 重置缩放" << endl;
        cout << "Q - 退出程序" << endl;
        cout << "====================" << endl;

        runSingleImageMode();
    }

    void loadCurrentImage() 
    {
        if (imageFiles.empty()) return;

        currentImage = imread(imageFiles[currentIndex], IMREAD_COLOR);
        if (currentImage.empty()) { cout << "无法加载图片: " << imageFiles[currentIndex] << endl; return; }

        currentImage.copyTo(originalImage); currentImage.copyTo(displayImage);
        scale = 1.0; updateDisplay();
    }

    void initializeCropRect() 
    {
        if (originalImage.empty()) return;

        int imgWidth = originalImage.cols;
        int imgHeight = originalImage.rows;

        // 如果图片比目标尺寸小，则调整目标尺寸
        if (imgWidth < targetWidth || imgHeight < targetHeight) 
        {
            cout << "警告: 图片尺寸 " << imgWidth << "x" << imgHeight
                << " 小于目标尺寸 " << targetWidth << "x" << targetHeight << "，将调整目标尺寸" << endl;

            // 保持宽高比，调整到最大可能尺寸
            float imgRatio = static_cast<float>(imgWidth) / imgHeight;
            float targetRatio = static_cast<float>(targetWidth) / targetHeight;

            if (imgRatio > targetRatio) 
            {
                targetHeight = imgHeight;
                targetWidth = static_cast<int>(imgHeight * targetRatio);
            }
            else 
            {
                targetWidth = imgWidth;
                targetHeight = static_cast<int>(imgWidth / targetRatio);
            }

            // 确保是64的倍数
            targetWidth = 64 * static_cast<int>(round(static_cast<float>(targetWidth) / 64));
            targetHeight = 64 * static_cast<int>(round(static_cast<float>(targetHeight) / 64));

            cout << "调整后目标尺寸: " << targetWidth << "x" << targetHeight << endl;
        }

        // 计算初始位置（居中）- 在显示坐标空间
        int displayWidth = static_cast<int>(imgWidth * scale);
        int displayHeight = static_cast<int>(imgHeight * scale);
        int x = (displayWidth - targetWidth) / 2;
        int y = (displayHeight - targetHeight) / 2;

        // 确保在显示范围内
        x = max(0, min(x, displayWidth - targetWidth));
        y = max(0, min(y, displayHeight - targetHeight));

        cropRect = Rect(x, y, targetWidth, targetHeight);

        cout << "初始化裁剪区域: (" << cropRect.x << ", " << cropRect.y << ") - ("
            << (cropRect.x + cropRect.width) << ", " << (cropRect.y + cropRect.height) << ")" << endl;
    }

    void updateDisplay() 
    {
        if (originalImage.empty()) return;

        Mat displayCopy;

        if (abs(scale - 1.0) < 0.01) originalImage.copyTo(displayCopy);// 无缩放，直接使用原图
        else resize(originalImage, displayCopy, Size(), scale, scale, INTER_LANCZOS4);// 应用缩放

        // 绘制固定大小的选择框
        rectangle(displayCopy, cropRect, Scalar(0, 0, 255), 3);

        // 在框内显示半透明覆盖
        Mat overlay;
        displayCopy.copyTo(overlay);
        if (cropRect.x >= 0 && cropRect.y >= 0 && cropRect.x + cropRect.width <= overlay.cols &&
            cropRect.y + cropRect.height <= overlay.rows) 
        {
            Mat roi = overlay(cropRect);
            Mat color(roi.size(), roi.type(), Scalar(0, 0, 255));
            addWeighted(color, 0.2, roi, 0.8, 0, roi);
        }
        overlay.copyTo(displayCopy);

        // 显示信息
        string info = "[" + to_string(currentIndex + 1) + "/" + to_string(imageFiles.size()) + "] " +
            fs::path(imageFiles[currentIndex]).filename().string();
        putText(displayCopy, info, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);

        string sizeInfo = "目标尺寸: " + to_string(targetWidth) + "x" + to_string(targetHeight);
        putText(displayCopy, sizeInfo, Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);

        string zoomInfo = "缩放: " + to_string(static_cast<int>(scale * 100)) + "%";
        putText(displayCopy, zoomInfo, Point(10, 90), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);

        string coordInfo = "位置: (" + to_string(cropRect.x) + ", " + to_string(cropRect.y) + ")";
        putText(displayCopy, coordInfo, Point(10, 120), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);

        string help = "拖拽移动选择框 | 滚轮缩放图片(框大小不变) | S:保存替换原图 | D:删除 | N:下一张 | P:上一张 | R:重置缩放 | Q:退出";
        putText(displayCopy, help, Point(10, displayCopy.rows - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);

        displayCopy.copyTo(displayImage);
        imshow("LoRA图片预处理工具 - 逐张处理模式", displayCopy);
    }

    void mouseCallback(int event, int x, int y, int flags) 
    {
        if (event == EVENT_LBUTTONDOWN) 
        {
            if (cropRect.contains(Point(x, y))) { isDragging = true; dragStart = Point(x - cropRect.x, y - cropRect.y); }
        }
        else if (event == EVENT_MOUSEMOVE && isDragging) 
        {
            // 计算新的框位置
            int newX = x - dragStart.x;
            int newY = y - dragStart.y;

            // 限制在显示范围内
            int displayWidth = static_cast<int>(originalImage.cols * scale);
            int displayHeight = static_cast<int>(originalImage.rows * scale);
            newX = max(0, min(newX, displayWidth - targetWidth));
            newY = max(0, min(newY, displayHeight - targetHeight));

            cropRect.x = newX;
            cropRect.y = newY;

            updateDisplay();
        }
        else if (event == EVENT_LBUTTONUP) isDragging = false;
        else if (event == EVENT_MOUSEWHEEL) 
        {
            // 处理鼠标滚轮事件 - 每次5%
            double scaleFactor = 1.05;

            if (getMouseWheelDelta(flags) > 0) { scale *= scaleFactor; if (scale > 5.0) scale = 5.0; }
            else { scale /= scaleFactor; if (scale < 0.1) scale = 0.1; }

            // 重新计算裁剪区域位置，保持相对位置
            double oldScale = scale / scaleFactor;
            if (oldScale > 0) 
            {
                cropRect.x = static_cast<int>(cropRect.x * scale / oldScale);
                cropRect.y = static_cast<int>(cropRect.y * scale / oldScale);
            }

            cout << "缩放比例: " << static_cast<int>(scale * 100) << "%" << endl;  updateDisplay();
        }
    }

    void saveCrop() 
    {
        if (originalImage.empty()) return;

        try 
        {
            // 先缩放图片到当前显示尺寸
            Mat scaledImage;
            resize(originalImage, scaledImage, Size(), scale, scale, INTER_LANCZOS4);

            // 验证裁剪区域
            if (cropRect.x < 0 || cropRect.y < 0 || cropRect.x + cropRect.width > scaledImage.cols ||
                cropRect.y + cropRect.height > scaledImage.rows) 
            {
                cout << "错误: 裁剪区域超出图像范围!" << endl;
                cout << "缩放后图像尺寸: " << scaledImage.cols << "x" << scaledImage.rows << endl;
                cout << "裁剪区域: (" << cropRect.x << ", " << cropRect.y << ") - ("
                    << (cropRect.x + cropRect.width) << ", " << (cropRect.y + cropRect.height) << ")" << endl;
                return;
            }

            // 从缩放后的图片中裁剪
            Mat cropped = scaledImage(cropRect);

            // 调整到目标尺寸
            Mat resized; resize(cropped, resized, Size(targetWidth, targetHeight), 0, 0, INTER_LANCZOS4);

            // 直接替换原图
            string originalPath = imageFiles[currentIndex];

            // 保存图片
            vector<int> compression_params;
            fs::path inputPath(originalPath);
            if (inputPath.extension() == ".jpg" || inputPath.extension() == ".jpeg") compression_params = { IMWRITE_JPEG_QUALITY, 95 };
            else if (inputPath.extension() == ".png") compression_params = { IMWRITE_PNG_COMPRESSION, 3 };

            if (imwrite(originalPath, resized, compression_params)) {
                cout << "成功保存并替换原图: " << originalPath << endl;
                cout << "在缩放 " << static_cast<int>(scale * 100) << "% 的图像上裁剪区域: ("
                    << cropRect.x << ", " << cropRect.y << ") - ("
                    << (cropRect.x + cropRect.width) << ", " << (cropRect.y + cropRect.height) << ")" << endl;

                // 重新加载更新后的图片
                loadCurrentImage(); nextImage();
            }
            else cout << "保存失败: " << originalPath << endl;
        }
        catch (const exception& e) {
            cout << "保存时出错: " << e.what() << endl;
            cout << "原始图像尺寸: " << originalImage.cols << "x" << originalImage.rows << endl;
            cout << "缩放比例: " << scale << endl;
            cout << "裁剪区域: (" << cropRect.x << ", " << cropRect.y << ") - ("
                << (cropRect.x + cropRect.width) << ", " << (cropRect.y + cropRect.height) << ")" << endl;
        }
    }

    void resetZoom() 
    {
        // 保存当前裁剪区域在原始图像中的对应位置
        if (scale > 0) {
            int origX = static_cast<int>(cropRect.x / scale);
            int origY = static_cast<int>(cropRect.y / scale);
            cropRect.x = origX; cropRect.y = origY;
        }

        scale = 1.0;
        cout << "缩放已重置" << endl;
        updateDisplay();
    }

    void deleteCurrentImage() 
    {
        if (imageFiles.empty()) return;

        string currentFile = imageFiles[currentIndex];

        try 
        {
            if (fs::remove(currentFile)) 
            {
                cout << "已删除图片: " << currentFile << endl;

                imageFiles.erase(imageFiles.begin() + currentIndex);

                if (imageFiles.empty()) 
                {
                    cout << "所有图片已处理完毕!" << endl;
                    destroyAllWindows(); return;
                }

                if (currentIndex >= imageFiles.size()) currentIndex = imageFiles.size() - 1;
                loadCurrentImage(); initializeCropRect();
            }
            else cout << "删除失败: " << currentFile << endl;
        }
        catch (const exception& e) 
        {
            cout << "删除时出错: " << e.what() << endl;
        }
    }

    void nextImage() 
    {
        if (currentIndex < imageFiles.size() - 1) 
        {
            currentIndex++;
            loadCurrentImage();
            initializeCropRect();
        }
        else cout << "已经是最后一张图片" << endl;
    }

    void previousImage() 
    {
        if (currentIndex > 0) 
        {
            currentIndex--;
            loadCurrentImage();
            initializeCropRect();
        }
    }

    void runSingleImageMode() 
    {
        while (true) 
        {
            updateDisplay();
            int key = waitKey(10) & 0xFF;

            switch (key) 
            {
            case 's':
            case 'S':
                saveCrop();
                break;
            case 'd':
            case 'D':
                deleteCurrentImage();
                break;
            case 'n':
            case 'N':
                nextImage();
                break;
            case 'p':
            case 'P':
                previousImage();
                break;
            case 'r':
            case 'R':
                resetZoom();
                break;
            case 'q':
            case 'Q':
                destroyAllWindows();
                return;
            }

            if (imageFiles.empty()) 
            {
                cout << "所有图片已处理完毕!" << endl;
                cv::destroyAllWindows(); break;
            }
        }
    }

    void batchProcess() 
    {
        if (imageFiles.empty()) return;

        cout << "\n=== 批量处理模式 ===" << endl;
        cout << "将处理 " << imageFiles.size() << " 张图片" << endl;
        cout << "目标尺寸: " << targetWidth << "x" << targetHeight << endl;
        cout << "使用居中裁剪方式，直接替换原图" << endl;
        cout << "开始处理? (y/n): ";

        char confirm;
        cin >> confirm;

        if (confirm != 'y' && confirm != 'Y') 
        {
            cout << "取消批量处理" << endl; return;
        }

        int processed = 0;
        int skipped = 0;

        for (size_t i = 0; i < imageFiles.size(); ++i) 
        {
            Mat img = imread(imageFiles[i], IMREAD_COLOR);
            if (img.empty()) 
            {
                cout << "跳过无法读取的图片: " << imageFiles[i] << endl;
                skipped++;
                continue;
            }

            try 
            {
                int currentTargetWidth = targetWidth;
                int currentTargetHeight = targetHeight;

                if (img.cols < targetWidth || img.rows < targetHeight) 
                {
                    float imgRatio = static_cast<float>(img.cols) / img.rows;
                    float targetRatio = static_cast<float>(targetWidth) / targetHeight;

                    if (imgRatio > targetRatio) 
                    {
                        currentTargetHeight = img.rows;
                        currentTargetWidth = static_cast<int>(img.rows * targetRatio);
                    }
                    else 
                    {
                        currentTargetWidth = img.cols;
                        currentTargetHeight = static_cast<int>(img.cols / targetRatio);
                    }

                    currentTargetWidth = 64 * static_cast<int>(round(static_cast<float>(currentTargetWidth) / 64));
                    currentTargetHeight = 64 * static_cast<int>(round(static_cast<float>(currentTargetHeight) / 64));
                }

                Rect centerCrop = calculateCenterCrop(img, currentTargetWidth, currentTargetHeight);
                Mat cropped = img(centerCrop);
                Mat resized;
                resize(cropped, resized, Size(currentTargetWidth, currentTargetHeight), 0, 0, INTER_LANCZOS4);

                string originalPath = imageFiles[i];
                vector<int> compression_params;
                fs::path inputPath(originalPath);
                if (inputPath.extension() == ".jpg" || inputPath.extension() == ".jpeg") compression_params = { IMWRITE_JPEG_QUALITY, 95 };
                else if (inputPath.extension() == ".png") compression_params = { IMWRITE_PNG_COMPRESSION, 3 };

                if (imwrite(originalPath, resized, compression_params)) 
                {
                    processed++;
                    cout << "处理完成: " << originalPath << " (" << i + 1 << "/" << imageFiles.size() << ")" << endl;
                }
                else 
                {
                    skipped++;
                    cout << "保存失败: " << originalPath << endl;
                }

            }
            catch (const exception& e) 
            {
                cout << "处理图片时出错 " << imageFiles[i] << ": " << e.what() << endl;
                skipped++;
            }
        }

        cout << "批量处理完成! 成功: " << processed << ", 失败: " << skipped << endl;
    }

    Rect calculateCenterCrop(const Mat& img, int cropWidth, int cropHeight) 
    {
        int imgWidth = img.cols;
        int imgHeight = img.rows;
        float targetRatio = static_cast<float>(cropWidth) / cropHeight;
        float imgRatio = static_cast<float>(imgWidth) / imgHeight;

        int finalCropWidth, finalCropHeight, x, y;

        if (imgRatio > targetRatio) 
        {
            finalCropHeight = imgHeight;
            finalCropWidth = static_cast<int>(imgHeight * targetRatio);
            x = (imgWidth - finalCropWidth) / 2;
            y = 0;
        }
        else 
        {
            finalCropWidth = imgWidth;
            finalCropHeight = static_cast<int>(imgWidth / targetRatio);
            x = 0;
            y = (imgHeight - finalCropHeight) / 2;
        }

        return Rect(x, y, finalCropWidth, finalCropHeight);
    }

    void run() 
    {
        if (imageFiles.empty()) 
        {
            cout << "当前文件夹中没有找到支持的图片文件!" << endl;
            cout << "支持的格式: ";
            for (const auto& fmt : supportedFormats) cout << fmt << " ";
            cout << endl; return;
        }

        selectProcessingMode();
    }
};

int main() 
{
    try 
    {
        ImageProcessor processor;
        processor.run();
    }
    catch (const exception& e) 
    {
        cerr << "程序出错: " << e.what() << endl;
        return 1;
    }
}