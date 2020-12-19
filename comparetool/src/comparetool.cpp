#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace std;
using namespace cv;

ushort find_segments(const Mat &image, Mat &labels)
{
    const int dims = 3;
    const int size[] = {256, 256, 256};
    SparseMat sparse(dims, size, CV_16U);
    
    ushort unique = 0;

    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            Vec3b color = image.at<Vec3b>(i, j);
            ushort &index = sparse.ref<ushort>(color[0], color[1], color[2]);
            
            if (index == 0)
            {
                unique++;
                index = unique;
            }

            labels.at<ushort>(i, j) = (index - 1);
        }
    }

    return unique;
}

Mat mask(const Mat& labels, ushort id)
{
    Mat out = Mat(labels.rows, labels.cols, CV_16UC1);

    for (int i = 0; i < labels.rows; i++)
    {
        for (int j = 0; j < labels.cols; j++)
        {
            out.at<ushort>(i, j) = (ushort)(labels.at<ushort>(i, j) == id);
        }
    }

    return out;
}

Mat *make_masks(const Mat& labels, int N)
{
    Mat *out = new Mat[N];

    for (size_t i = 0; i < N; i++)
    {
        out[i] = mask(labels, i);
    }

    return out;
}

void score(const Mat *in_masks, int in_N, const Mat *gt_masks, int gt_N)
{
    size_t asa_total = 0;
    size_t use_total = 0;

    for (int inn = 0; inn < in_N; inn++)
    {
        size_t intermax = 0;
        for (int gtn = 0; gtn < gt_N; gtn++)
        {
            size_t intersect = sum(in_masks[inn] & gt_masks[gtn]).val[0];
            size_t diff = sum(in_masks[inn] & ~gt_masks[gtn]).val[0];

            if (intersect > intermax)
            {
                intermax = intersect;
            }

            use_total += (intersect < diff) ? intersect : diff;
        }

        asa_total += intermax;
    }

    cout << (double)(asa_total) / (gt_masks[0].rows * gt_masks[0].cols) << endl;
    cout << (double)(use_total) / (gt_masks[0].rows * gt_masks[0].cols) << endl;
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        cout << "Usage: comparetool INPUT GROUND_TRUTH" << endl;
        return 1;
    }

    Mat in_image = imread(argv[1]);
    Mat gt_image = imread(argv[2]);

    if (in_image.empty() || gt_image.empty())
    {
        cout << "Failed to load INPUT and/or GROUND_TRUTH." << endl;
        return 1;
    }

    Mat in_labels(in_image.rows, in_image.cols, CV_16UC1);
    Mat gt_labels(in_image.rows, in_image.cols, CV_16UC1);

    ushort in_N = find_segments(in_image, in_labels);
    ushort gt_N = find_segments(gt_image, gt_labels);
    
    cout << in_N << endl;
    cout << gt_N << endl;

    Mat *in_masks = make_masks(in_labels, in_N);
    Mat *gt_masks = make_masks(gt_labels, gt_N);

    score(in_masks, in_N, gt_masks, gt_N);
    
    return 0;
}
