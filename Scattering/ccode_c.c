/* Copyright (c) 2020 Vladyslav Andriiashen
 * Centrum Wiskunde & Informatica, Amsterdam, the Netherlands.
 *
 * Code is available via AppleCT Dataset Project; www.github.com/cicwi/applect-dataset-project
 *
 * Referenced paper: S.B. Coban, V. Andriiashen, P.S. Ganguly, et al.
 * Parallel-beam X-ray CT datasets of apples with internal defects and label balancing for machine learning. 2020. www.arxiv.org/abs/2012.13346
 *
 *Dataset available via Zenodo; 10.5281/zenodo.4212301
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

// Compute scattering distribution for an image.
// The image size is given by height and width.
// Pixel size is given by px_size and used to convert distance between pixels to mm (blur sigma is given in mm).
// Arrays A, B, sigma1 and sigma2 are of the same size as the image and contain blur parameters for every pixels
void opt_array_scattering(const int height, const int width, const float px_size,
                          float* const A, float* const B, float* const sigma1, float* const sigma2, float* tmp)
{
    // Allocate temporary arrays
    float * dxt = (float *) malloc(width*width*sizeof(float));
    float * dyt = (float *) malloc(height*height*sizeof(float));
    float * s1t = (float *) malloc(width*height*sizeof(float));
    float * s2t = (float *) malloc(width*height*sizeof(float));
    const float px2 = px_size*px_size;
    
    // Compute distances between all pairs of pixels in the image.
    // dyt - vertical coordinate of the distance.
    #pragma omp parallel for
    for(int j = 0; j < height;j++){
        for(int src_j = 0; src_j < j; src_j++){
            dyt[j*height + src_j] = (j - src_j)*(j - src_j) * px2;
            dyt[src_j*height + j] = dyt[j*height + src_j];
        }
        dyt[j*height + j] = 0;
    }
    // dxt - horizontal coordinate of the distance.
    #pragma omp for
    for(int i = 0; i < width; i++){
        for(int src_i = 0; src_i < i; src_i++) {
            dxt[i*width + src_i] = (i - src_i)*(i - src_i) * px2;
            dxt[src_i*width + i] = dxt[i*width + src_i];
        }
        dxt[i*width + i] = 0;
    }
    // Fill square values of sigma that will be used later
    #pragma omp parallel for
    for(int i = 0; i < height*width; i++) {
        s1t[i] = 2*sigma1[i]*sigma1[i];
        s2t[i] = 2*sigma2[i]*sigma2[i];
    }
    
    // Two outermost loops iterate over target pixels
    #pragma omp parallel for
    for(int j = 0; j < height; j++) {
        for(int i = 0; i < width; i++) {
            // tmpval accumulate scattering contribution from all pixels of the images
            float tmpval=0;
            // Initialze pointer positions for values in arrays
            float * curA = A;
            float * curB = B;
            float * curs1 = s1t;
            float * curs2 = s2t;
            float * dy = dyt + j*height;

            // Start to iterate over all pixels of the image that contribute to the current target pixel
            for(int src_j = 0; src_j < height; src_j++) {
                float * dx = dxt + i*width;
                
                for(int src_i = 0; src_i < width; src_i++) {
                    // tmpval is updated according to the empirical formula
                    tmpval += (*curA) * expf(-(*dx + *dy) / (*curs1));
                    tmpval += (*curB) * expf(-(*dx + *dy) / (*curs2));
                    // Pointers are shifted explicitly
                    curA++;
                    curB++;
                    curs1++;
                    curs2++;
                    dx++;
                }
                
                dy++;
            }
            // After two loops iterating over source pixels the value of tmpval is saved in the result array.
            tmp[j*width + i] = tmpval;
        }
    }
    
    // Memory deallocation for temporary arrays
    free(dxt);
    free(dyt);
    free(s1t);
    free(s2t);
}

// Compute scattering distribution for a range of rows in the image. This function is similar to opt_array_scattering.
// The image size is given by height and width.
// Pixel size is given by px_size and used to convert distance between pixels to mm (blur sigma is given in mm).
// Arrays A, B, sigma1 and sigma2 are of the same size as the image and contain blur parameters for every pixels.
// Range of rows is given by row_start and row_end.
void target_row_scattering(const int row_start, const int row_end, const int height, const int width, const float px_size,
                          float* const A, float* const B, float* const sigma1, float* const sigma2, float* tmp)
{
    // Allocate temporary arrays
    const int row_range = row_end - row_start;
    float * dxt = (float *) malloc(width*width*sizeof(float));
    float * dyt = (float *) malloc(row_range*height*sizeof(float));
    float * s1t = (float *) malloc(width*height*sizeof(float));
    float * s2t = (float *) malloc(width*height*sizeof(float));
    const float px2 = px_size*px_size;
    
    // Compute distances between all pairs of pixels in the image.
    // dyt - vertical coordinate of the distance.
    #pragma omp parallel for
    for(int j = row_start; j < row_end; j++){
        for(int src_j = 0; src_j < height; src_j++) {
            dyt[(j-row_start)*height + src_j] = (j - src_j)*(j - src_j) * px2;
        }
    }
    // dxt - horizontal coordinate of the distance.
    #pragma omp for
    for(int i = 0; i < width; i++){
        for(int src_i = 0; src_i < i; src_i++) {
            dxt[i*width + src_i] = (i - src_i)*(i - src_i) * px2;
            dxt[src_i*width + i] = dxt[i*width + src_i];
        }
        dxt[i*width + i] = 0;
    }
    // Fill square values of sigma that will be used later
    #pragma omp parallel for
    for(int i = 0; i < height*width; i++) {
        s1t[i] = 2*sigma1[i]*sigma1[i];
        s2t[i] = 2*sigma2[i]*sigma2[i];
    }
    
    // Two outermost loops iterate over target pixels
    // Row of the target pixel is in the [row_start, row_end] range.
    for(int j = row_start; j < row_end; j++) {
    	#pragma omp parallel for
        for(int i = 0; i < width; i++) {
            // tmpval accumulate scattering contribution from all pixels of the images
            float tmpval=0;
            // Initialze pointer positions for values in arrays
            float * curA = A;
            float * curB = B;
            float * curs1 = s1t;
            float * curs2 = s2t;
            float * dy = dyt;

            // Start to iterate over all pixels of the image that contribute to the current target pixel
            for(int src_j = 0; src_j < height; src_j++) {
                float * dx = dxt + i*width;
                    
                for(int src_i = 0; src_i < width; src_i++) {
                    // tmpval is updated according to the empirical formula
                    tmpval += (*curA) * expf(-(*dx + *dy) / (*curs1));
                    tmpval += (*curB) * expf(-(*dx + *dy) / (*curs2));
                    // Pointers are shifted explicitly
                    curA++;
                    curB++;
                    curs1++;
                    curs2++;
                    dx++;
                }
                    
                dy++;
            }
            // After two loops iterating over source pixels the value of tmpval is saved in the result array.
            tmp[(j-row_start)*width + i] = tmpval;
        }
    }
    
    // Memory deallocation for temporary arrays
    free(dxt);
    free(dyt);
    free(s1t);
    free(s2t);
}
