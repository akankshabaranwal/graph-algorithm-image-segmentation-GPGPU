#include <iostream>

// Libraries for reading & writing images, don't change!
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"

#include "gaussian_filter.h"

// Get R, G and B channels from image, ignoring A component.
void load_image(const char* path, int &width, int& height, int& channels, unsigned char **img) {
    channels = 3; // For now only color images!
    *img = stbi_load(path, &width, &height, nullptr, 3);
    if(*img == nullptr) {
        printf("Error in loading the image\n");
        exit(1);
    }
}

int clamp(int val, int min, int max)
{
    return std::min(std::max(val, min), max);
}

// Using colorimetric conversion (https://en.wikipedia.org/wiki/Grayscale)
double* grayscale(unsigned char* image, int w, int h)
{
    double* out = new double[w * h];

    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            int r = image[(y * w + x) * 3];
            int g = image[(y * w + x) * 3 + 1];
            int b = image[(y * w + x) * 3 + 2];

            out[y * w + x] = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
        }
    }

    return out; // To-do: Return smart pointer so we don't need to delete[]
}

// Re-scale double matrix to (0, 1). Used after applying a filter.
double* normalize(double* image, int w, int h)
{
    double* out = new double[w * h];

    double min = INFINITY;
    double max = -INFINITY;

    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            min = std::min(image[y * w + x], min);
            max = std::max(image[y * w + x], max);
        }
    }

    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            out[y * w + x] = (image[y * w + x] - min) / (max - min);
        }
    }

    return out; // To-do: Return smart pointer so we don't need to delete[]
}

// Convert double matrix to single-channel image
unsigned char* convert(double* image, int w, int h)
{
    unsigned char* out = new unsigned char[w * h];

    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            out[y * w + x] = (unsigned char)(clamp(image[y * w + x] * 255, 0, 255));
        }
    }

    return out; // To-do: Return smart pointer so we don't need to delete[]
}

int main() {
    const char* in_path = "data/beach.gif";
    const char* out_path = "data/result.png";

    int width, height, channels;
    unsigned char *img;
    load_image(in_path, width, height, channels, &img);

    printf("Loaded image with a width of %dpx, a height of %dpx\n", width, height);

    double* gray = grayscale(img, width, height);

    double* blurred = new double[width * height];
    double* filter = gaussian_filter(5, 5, 2.0);
    convolve2d(gray, blurred, width, height, filter, 5, 5);
    double* normal = normalize(blurred, width, height);
    unsigned char* out = convert(normal, width, height);

    stbi_write_png(out_path, width, height, 1, out, width);
    stbi_image_free(img);

    // To-do: Cleanup using delete[] on alloc'ed images/matrices

    return 0;
}
