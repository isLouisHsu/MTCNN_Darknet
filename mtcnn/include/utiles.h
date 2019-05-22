#ifndef __UTILES_H
#define __UTILES_H

#include <opencv/cv.h>
#include "darknet.h"


IplImage *image_to_ipl(image im);
image     ipl_to_image(IplImage* src);

void  show_ipl(IplImage* ipl, char* winname, int pause);
void  show_im(image im, char* winname, int pause);

image rgb_to_bgr(image im);
image resize_image_scale(image im, float scale);

static inline float _min(float a, float b){return a<b? a: b;}
static inline float _max(float a, float b){return a>b? a: b;}
static inline int _ascending(const void * a, const void * b){return ( *(int*)a - *(int*)b );}
static inline int _descending(const void * a, const void * b){return ( *(int*)b - *(int*)a );}

void find_max_min(float* x, int n, float* max, float* min);

#endif