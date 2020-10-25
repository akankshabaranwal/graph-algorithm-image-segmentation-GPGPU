#include <iostream>

// Libraries for reading & writing images, don't change!
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"

// Get R, G and B channels from image, ignoring A component.
void load_image(const char* path, int &width, int& height, int& channels, unsigned char **img) {
    channels = 3; // For now only color images!
    *img = stbi_load(path, &width, &height, nullptr, 3);
    if(*img == nullptr) {
        printf("Error in loading the image\n");
        exit(1);
    }
}

int main() {
    const char* in_path = "data/beach.gif";
    const char* out_path = "data/result.png";

    int width, height, channels;
    unsigned char *img;
    load_image(in_path, width, height, channels, &img);

    printf("Loaded image with a width of %dpx, a height of %dpx\n", width, height);

    stbi_write_png(out_path, width, height, channels, img, width * channels);
    stbi_image_free(img);

    return 0;
}
