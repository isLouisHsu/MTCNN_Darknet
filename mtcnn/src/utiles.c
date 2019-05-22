#include "utiles.h"

IplImage *image_to_ipl(image im)
{
    int x,y,c;
    IplImage *disp = cvCreateImage(cvSize(im.w,im.h), IPL_DEPTH_8U, im.c);
    int step = disp->widthStep;
    for(y = 0; y < im.h; ++y){
        for(x = 0; x < im.w; ++x){
            for(c= 0; c < im.c; ++c){
                float val = im.data[c*im.h*im.w + y*im.w + x];
                disp->imageData[y*step + x*im.c + c] = (unsigned char)(val*255);
            }
        }
    }
    return disp;
}

image ipl_to_image(IplImage* src)
{
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    image im = make_image(w, h, c);
    unsigned char *data = (unsigned char *)src->imageData;
    int step = src->widthStep;
    int i, j, k;

    for(i = 0; i < h; ++i){
        for(k= 0; k < c; ++k){
            for(j = 0; j < w; ++j){
                im.data[k*w*h + i*w + j] = data[i*step + j*c + k]/255.;
            }
        }
    }
    return im;
}

void show_ipl(IplImage* ipl, char* winname, int pause)
{
    cvNamedWindow(winname);
    cvShowImage(winname, ipl);
    cvWaitKey(pause);
    cvDestroyWindow(winname);
}

void show_im(image im, char* winname, int pause)
{
    IplImage* ipl = image_to_ipl(im);
    show_ipl(ipl, winname, pause);
}

image rgb_to_bgr(image im)
{
    IplImage* ipl = image_to_ipl(im);
    cvCvtColor(ipl, ipl, CV_BGR2RGB);
    im = ipl_to_image(ipl);

    return im;
}

image resize_image_scale(image im, float scale)
{
    int wn = (int)(im.w * scale);
    int hn = (int)(im.h * scale);

#if 1
    image sized = resize_image(im, wn, hn);
    return sized;
#else
    IplImage* tmp = image_to_ipl(im);
    IplImage* sized = cvCreateImage(cvSize(wn, hn), tmp->depth, tmp->nChannels);
    cvResize(tmp, sized, CV_INTER_LINEAR);
    return ipl_to_image(sized);
#endif
}

void find_max_min(float* x, int n, float* max, float* min)
{
    *max = -99999.;
    *min = 99999.;
    for (int i = 0; i < n; i++ ){
        float val = x[i];
        if (val > *max) *max = val;
        if (val < *min) *min = val;
    }
}
