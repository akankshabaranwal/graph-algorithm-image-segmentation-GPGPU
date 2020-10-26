//
// Created by Amory Hoste on 25/10/2020.
//

#define _USE_MATH_DEFINES
#include <cmath>

#include "gaussian_filter.h"

double* gaussian_filter(int w, int h, double sigma)
{
	double* filter = new double[w * h];
	double scale = 1.0 / (2 * sigma * sigma);

	for (int y = 0; y < h; y++)
	{
		for (int x = 0; x < w; x++)
		{
			int mx = x - w / 2;
			int my = y - h / 2;

			filter[y * w + x] = (1.0 / M_PI) * scale * exp(-scale * mx * mx * my * my);
		}
	}

	return filter; // To-do: Return smart pointer so we don't need to delete[]
}

// Applies a single convolve step using (cx, cy) as the center
void apply2d(double* in, double* out, int inw, int inh, double* filter, int fw, int fh, int cx, int cy)
{
	double sum = 0.0;

	for (int fy = 0; fy < fh; fy++)
	{
		for (int fx = 0; fx < fw; fx++)
		{
			int absx = cx + fx - (fw / 2);
			int absy = cy + fy - (fh / 2);
			int ffx = fw - fx - 1;
			int ffy = fh - fy - 1;

			if ((0 <= absx && absx < inw) && (0 <= absy && absy < inh))
			{
				sum += in[absy * inw + absx] * filter[ffy * fw + ffx];
			}
		}
	}

	out[cy * inw + cx] = sum;
}

void convolve2d(double* in, double* out, int inw, int inh, double* filter, int fw, int fh)
{
	for (int cy = 0; cy < inh; cy++)
	{
		for (int cx = 0; cx < inw; cx++)
		{
			apply2d(in, out, inw, inh, filter, fw, fh, cx, cy);
		}
	}
}
