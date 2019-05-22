#ifndef __MTCNN_H
#define __MTCNN_H

#include <opencv/cv.h>
#include <opencv2/videoio/videoio_c.h>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgcodecs/imgcodecs_c.h>
#include <math.h>
#include "darknet.h"
#include "list.h"
#include "image.h"
#include "activations.h"
#include "utiles.h"

// #define LIST 1

typedef struct bbox
{
    float x1, y1, x2, y2;   /* 左上、右下 */
} bbox;

typedef struct landmark
{
    float x1, y1, x2, y2, x3, y3, x4, y4, x5, y5; /* 左眼、右眼、鼻尖、左嘴角、右嘴角 */
} landmark;

typedef struct detect
{
    float score;    /* 该框评分 */
    bbox bx;        /* 回归方框 */
    bbox offset;    /* 偏置 */
    landmark mk;    /* 位置 */
} detect;

typedef struct params
{
    float min_face;     /* minimal face size */
    float pthresh;      /* threshold of pnet */
    float rthresh;      /* threshold of rnet */
    float othresh;      /* threshold of onet */
    float scale;        /* scale factor */
    int stride;         
    int cellsize;       /* size of cell */
} params;

box bbox_to_box(bbox bbx);
float bbox_area(bbox a);
float bbox_iou(bbox a, bbox b, int mode);

network* load_mtcnn_net(char* netname);
params initParams(int argc, char **argv);
void run_image(int argc, char **argv);
void run_video(int argc, char **argv);

#ifndef LIST
void detect_image(network *pnet, network *rnet, network *onet, image im, int* n, detect** dets, params p);
void show_detect(image im, detect* dets, int n, char* winname, int pause, int showscore, int showbox, int showmark);
#else

#endif

#endif
