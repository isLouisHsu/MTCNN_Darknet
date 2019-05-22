#include "mtcnn.h"


// ================================================================================== //

box bbox_to_box(bbox bbx)
{
    box bx;
    bx.x = (bbx.x1 + bbx.x2) / 2;
    bx.y = (bbx.y1 + bbx.y2) / 2;
    bx.w = bbx.x2 - bbx.x1;
    bx.h = bbx.y2 - bbx.y1;
    return bx;
}

float bbox_overlap(float ax1, float ax2, float bx1, float bx2)
{
    float left  = _max(ax1, bx1);
    float right = _min(ax2, bx2);
    float gap = right - left;
    return gap;
}

float bbox_intersection(bbox a, bbox b)
{
    float w = bbox_overlap(a.x1, a.x2, b.x1, b.x2);
    float h = bbox_overlap(a.y1, a.y2, b.y1, b.y2);
    if(w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
}

float bbox_area(bbox a)
{
    float w = a.x2 - a.x1;
    float h = a.y2 - a.y1;
    return w*h;
}

float bbox_union(bbox a, bbox b, int mode)
{
    float u = 0;
    
    float area_a = bbox_area(a);
    float area_b = bbox_area(b);

    if (mode == 0){
        u = area_a + area_b - bbox_intersection(a, b);
    } else if (mode == 1){
        u = _min(area_a, area_b);
    } 

    return u;
}

float bbox_iou(bbox a, bbox b, int mode)
{
    float i = bbox_intersection(a, b);
    float u = bbox_union(a, b, mode);
    return i/u;
}

// ================================================================================== // 

params initParams(int argc, char **argv)
{
    params p;

    p.min_face = find_float_arg(argc, argv, "--minface", 96.0);
    p.pthresh  = find_float_arg(argc, argv, "-p", 0.8);
    p.rthresh  = find_float_arg(argc, argv, "-r", 0.8);
    p.othresh  = find_float_arg(argc, argv, "-o", 0.8);
    p.scale    = find_float_arg(argc, argv, "--scale", 0.79);
    p.stride   = find_int_arg(argc, argv, "--stride", 2);
    p.cellsize = find_int_arg(argc, argv, "--cellsize", 12);

    return p;
}

network* load_mtcnn_net(char* netname){
    char cfg[64];
    char weights[64];
    
    sprintf(cfg, "cfg/%s.cfg", netname);
    sprintf(weights, "weights/%s.weights", netname);
    
    return load_network(cfg, weights, 0);
}

#ifndef LIST

// ================================================================================== // 

// sorted in decending order
void bsort(detect** dets, int n)
{
    for (int i = 0; i < n - 1; i++ ){
        for (int j = 0; j < n - 1 - i; j++ ){
            float a = (*dets)[j].score;
            float b = (*dets)[j+1].score;

            if (a < b){
                detect tmp = (*dets)[j];
                (*dets)[j] = (*dets)[j+1];
                (*dets)[j+1] = tmp;
            }
        }
    }
}

void _nms(detect* dets, int n, float thresh, int mode)
{
    // DEBUG
//     float max = 0;
//     float min = 0;
//
//     max = -99999.;
//     min = 99999.;
//     for (int i = 0; i < n; i++ ){
//         float val = dets[i].score;
//         if (val > max) max = val;
//         if (val < min) min = val;
//     }

    bsort(&dets, n);

     // DEBUG
//     max = -99999.;
//     min = 99999.;
//     for (int i = 0; i < n; i++ ){
//         float val = dets[i].score;
//         if (val > max) max = val;
//         if (val < min) min = val;
//     }

    // do nms
    for (int i = 0; i < n; i++)
    {
        if (dets[i].score == 0) continue;
        bbox a = dets[i].bx;

        for (int j = i + 1; j < n; j++)
        {
            bbox b = dets[j].bx;

            if (bbox_iou(a, b, mode) > thresh){
                dets[j].score = 0;
            }
        }
    }
}

int _count_nzero(detect* dets, int n)
{
    int n_keep = 0;
    for (int i = 0; i < n; i++){
        float score = dets[i].score;
        if ((score == 0) || (isnan(score))) continue;
        n_keep += 1;
    }
    return n_keep;
}

void _cal_box(detect* dets, int n)
{
    for (int i = 0; i < n; i++ ){
        bbox box = dets[i].bx;
        float w = box.x2 - box.x1 + 1;
        float h = box.y2 - box.y1 + 1;

        bbox offset = dets[i].offset;
        box.x1 += offset.x1*w;
        box.y1 += offset.y1*h;
        box.x2 += offset.x2*w;
        box.y2 += offset.y2*h;

        dets[i].bx = box;
    }
}

void _cal_landmark(detect* dets, int n)
{
    for (int i = 0; i < n; i++ ){
        bbox box = dets[i].bx;
        float w = box.x2 - box.x1 + 1;
        float h = box.y2 - box.y1 + 1;

        landmark ldmark = dets[i].mk;
        ldmark.x1 += ldmark.x1*w + box.x1;
        ldmark.y1 += ldmark.y1*h + box.y1;
        ldmark.x2 += ldmark.x2*w + box.x1;
        ldmark.y2 += ldmark.y2*h + box.y1;
        ldmark.x3 += ldmark.x3*w + box.x1;
        ldmark.y3 += ldmark.y3*h + box.y1;
        ldmark.x4 += ldmark.x4*w + box.x1;
        ldmark.y4 += ldmark.y4*h + box.y1;
        ldmark.x5 += ldmark.x5*w + box.x1;
        ldmark.y5 += ldmark.y5*h + box.y1;

        dets[i].mk = ldmark;
    }
}

void _square(detect* dets, int n)
{
    for (int i = 0; i < n; i++ ){
        bbox box = dets[i].bx;
        float w = box.x2 - box.x1 + 1;
        float h = box.y2 - box.y1 + 1;

        float maxsize = _max(w, h);
        box.x1 += w/2 - maxsize/2;
        box.y1 += h/2 - maxsize/2;
        box.x2  = box.x1 + maxsize - 1;
        box.y2  = box.y1 + maxsize - 1;

        dets[i].bx = box;
    }
}

float* _crop_patch(image im, detect* dets, int n, float netsize)
{
    int size = netsize*netsize*im.c;
    float* X = calloc(n, sizeof(float)*size);
    int i, j, k, l;
    
    for (i = 0; i < n; i++ )
    {
        bbox a = dets[i].bx;

        int x1 = (int)a.x1; 
        int x2 = (int)a.x2;
        int y1 = (int)a.y1; 
        int y2 = (int)a.y2;

        int w = x2 - x1 + 1;
        int h = y2 - y1 + 1;

        int xx1 = 0;
        int yy1 = 0;
        int xx2 = w - 1;
        int yy2 = h - 1;

        if (x1 < 0){xx1 = - x1; x1 = 0;}
        if (y1 < 0){yy1 = - y1; y1 = 0;}
        if (x2 > im.w - 1){xx2 = w + im.w - x2 - 2; x2 = im.w - 1;}
        if (y2 > im.h - 1){yy2 = h + im.h - y2 - 2; y2 = im.h - 1;}

        // generate patch image
        image patch = make_image(w, h, im.c);
        for(j = 0; j < im.c; j++ ){
            for(k = yy1; k < yy2 + 1; k++ ){
                for(l = xx1; l < xx2 + 1; l++ ){
                    int x = x1 + l;
                    int y = y1 + k;
                    float val = im.data[x + y*im.w + j*im.w*im.h];
                    patch.data[l + k*w + j*w*h] = val;
                }
            }
        }

#if 0
        // DEBUG
        char buff[256];
        sprintf(buff, "/home/louishsu/Desktop/patches_c/%d", i);
        //        show_im(sized, buff, 500);
        save_image(patch, buff);
#endif

        // resize patch
        image sized = resize_image(patch, (int)netsize, (int)netsize);

        // copy patch
        memcpy(X + i*size, sized.data, sizeof(float)*size);

        free_image(patch);
        free_image(sized);
    }

    return X;
}

// ================================================================================== // 

void _detect_pnet(network *net, image im, int* n, detect** dets, params p)
{
    float NETSIZE = 12.;
    int n_boxes = 0;
    int i, j;

    float cur_scale = NETSIZE / p.min_face; 
    image cur_img = resize_image_scale(im, cur_scale);
    detect* all_boxes = calloc(0, sizeof(detect));

    while(_min(cur_img.h, cur_img.w) >= NETSIZE){

        // forward network
        resize_network(net, cur_img.w, cur_img.h);
        float* X = cur_img.data;
        network_predict(net, X);

        // generate bbox
        layer l = net->layers[net->n-1];
        activate_array(l.output, l.w*l.h, LOGISTIC);


        for (i = 0; i < l.h; i++){
            for (j = 0; j < l.w; j++){
                float score = l.output[j + i*l.w];
                if (score > p.pthresh){

                    n_boxes += 1;
                    all_boxes = realloc(all_boxes, n_boxes*sizeof(detect));
                    
                    // boxes
                    all_boxes[n_boxes-1].score = l.output[j + i*l.w];
                    all_boxes[n_boxes-1].bx.x1 = j*p.stride / cur_scale;
                    all_boxes[n_boxes-1].bx.y1 = i*p.stride / cur_scale;
                    all_boxes[n_boxes-1].bx.x2 = (j*p.stride + p.cellsize) / cur_scale;
                    all_boxes[n_boxes-1].bx.y2 = (i*p.stride + p.cellsize) / cur_scale;

                    // offsets
                    all_boxes[n_boxes-1].offset.x1 = l.output[j + i*l.w + 1*l.w*l.h];
                    all_boxes[n_boxes-1].offset.y1 = l.output[j + i*l.w + 2*l.w*l.h];
                    all_boxes[n_boxes-1].offset.x2 = l.output[j + i*l.w + 3*l.w*l.h];
                    all_boxes[n_boxes-1].offset.y2 = l.output[j + i*l.w + 4*l.w*l.h];
                }
            }
        }

        // update scale
        cur_scale *= p.scale;
        free_image(cur_img);
        cur_img = resize_image_scale(im, cur_scale);
    }
    free_image(cur_img);

    if (n_boxes == 0){
        *n = 0;
        *dets = realloc(*dets, 0);
        return;
    }

#if 0
    FILE *fp = fopen("/home/louishsu/Desktop/c_pnet_gen.txt", "w");
    char buff[256];

    bsort(&all_boxes, n_boxes);

    for (i = 0; i < n_boxes; i++ ){
        detect det = all_boxes[i];
        sprintf(buff, "%.6f %.6f %.6f %.6f %.6f\n",
                det.score, det.bx.x1, det.bx.y1, det.bx.x2, det.bx.y2);
        fputs(buff, fp);
    }
    fclose(fp);
#endif

    // nms
    _nms(all_boxes, n_boxes, 0.6, 0);

    // keep boxes
    j = 0;
    *n = _count_nzero(all_boxes, n_boxes);
    *dets = realloc(*dets, *n*sizeof(detect));
    for (int i = 0; i < n_boxes; i++ ){
        if (all_boxes[i].score != 0){
            (*dets)[j++] = all_boxes[i];
        }
    }
    free(all_boxes);

#if 0
    FILE *fp = fopen("/home/louishsu/Desktop/c_pnet_gen.txt", "w");
    char buff[256];

    bsort(dets, *n);

    for (i = 0; i < *n; i++ ){
        detect det = (*dets)[i];
        sprintf(buff, "%.6f %.6f %.6f %.6f %.6f\n",
                det.score, det.bx.x1, det.bx.y1, det.bx.x2, det.bx.y2);
        fputs(buff, fp);
    }
    fclose(fp);
#endif

    // refine boxes
     _cal_box(*dets, *n);
}

void _detect_rnet(network *net, image im, int* n, detect** dets, params p)
{
    if (*n == 0) return;

    int i;

    float NETSIZE = 24.;
    set_batch_network(net, *n);
    resize_network(net, (int)NETSIZE, (int)NETSIZE);

    // generate data
    _square(*dets, *n);
    float* X = _crop_patch(im, *dets, *n, NETSIZE);

    // forward network
    network_predict(net, X);
    free(X);

    layer l = net->layers[net->n-1];
    // update
    for (i = 0; i < *n; i++ ){

        float *output = l.output + i*l.outputs;

        float score = *(output + 0);
        score  = logistic_activate(score);

        if ((score == 0) || isnan(score) || (score < p.rthresh)){
            (*dets)[i].score = 0;
        } else {
            (*dets)[i].score = score;
        }

        (*dets)[i].offset.x1 = *(output + 1);
        (*dets)[i].offset.y1 = *(output + 2);
        (*dets)[i].offset.x2 = *(output + 3);
        (*dets)[i].offset.y2 = *(output + 4);
    }

#if 0
    FILE *fp = fopen("/home/louishsu/Desktop/c_pnet_gen.txt", "w");
    char buff[256];

    bsort(dets, *n);

    for (i = 0; i < *n; i++ ){
        detect det = (*dets)[i];
//        sprintf(buff, "%.6f %.6f %.6f %.6f %.6f\n",
//                det.score, det.bx.x1, det.bx.y1, det.bx.x2, det.bx.y2);
        sprintf(buff, "%.6f %.6f %.6f %.6f %.6f\n",
                det.score, det.offset.x1, det.offset.y1, det.offset.x2, det.offset.y2);
        fputs(buff, fp);
    }
    fclose(fp);
#endif

    // nms
    _nms(*dets, *n, 0.5, 0);
    int n_boxes = _count_nzero(*dets, *n);
    if (n_boxes == 0){
        *n = 0;
        *dets = realloc(*dets, 0);
        return;
    }

    // keep boxes
    int j = 0;
    detect* tmp = calloc(*n, sizeof(detect));
    memcpy(tmp, *dets, *n*sizeof(detect));          // store origin dets
    *dets = realloc(*dets, n_boxes*sizeof(detect)); // realloc dets

    for (i = 0; i < *n; i++ ){
        float score = tmp[i].score;
        if (score == 0) continue;
        (*dets)[j++] = tmp[i];
    }

    *n = n_boxes;
    free(tmp);

#if 0
    FILE *fp = fopen("/home/louishsu/Desktop/c_pnet_gen.txt", "w");
    char buff[512];

    bsort(dets, *n);

    for (i = 0; i < *n; i++ ){
        detect det = (*dets)[i];
        sprintf(buff, "%.6f %.6f %.6f %.6f %.6f\n",
                det.score, det.bx.x1, det.bx.y1, det.bx.x2, det.bx.y2);
        fputs(buff, fp);
    }
    fclose(fp);
#endif

    // refine boxes
     _cal_box(*dets, *n);
}

void _detect_onet(network *net, image im, int* n, detect** dets, params p)
{
    if (*n == 0) return;
    int i = 0;

    float NETSIZE = 48.;
    set_batch_network(net, *n);
    resize_network(net, (int)NETSIZE, (int)NETSIZE);

    // generate data
    _square(*dets, *n);
    float* X = _crop_patch(im, *dets, *n, NETSIZE);

    // forward network
    network_predict(net, X);
    free(X);

    layer l = net->layers[net->n-1];

    // update
    for (i = 0; i < *n; i++ ){

        float *output = l.output + i*l.outputs;

        float score = *(output + 0);
        score  = logistic_activate(score);

        if ((score == 0) || isnan(score) || (score < p.othresh)){
            (*dets)[i].score = 0;
        } else {
            (*dets)[i].score = score;
        }

        (*dets)[i].offset.x1 = *(output + 1);
        (*dets)[i].offset.y1 = *(output + 2);
        (*dets)[i].offset.x2 = *(output + 3);
        (*dets)[i].offset.y2 = *(output + 4);

        (*dets)[i].mk.x1 = *(output + 5);
        (*dets)[i].mk.y1 = *(output + 6);
        (*dets)[i].mk.x2 = *(output + 7);
        (*dets)[i].mk.y2 = *(output + 8);
        (*dets)[i].mk.x3 = *(output + 9);
        (*dets)[i].mk.y3 = *(output + 10);
        (*dets)[i].mk.x4 = *(output + 11);
        (*dets)[i].mk.y4 = *(output + 12);
        (*dets)[i].mk.x5 = *(output + 13);
        (*dets)[i].mk.y5 = *(output + 14);
    }


#if 0
    FILE *fp = fopen("/home/louishsu/Desktop/c_pnet_gen.txt", "w");
    char buff[256];

    bsort(dets, *n);

    for (i = 0; i < *n; i++ ){
        detect det = (*dets)[i];
//        sprintf(buff, "%.6f %.6f %.6f %.6f %.6f\n",
//                det.score, det.bx.x1, det.bx.y1, det.bx.x2, det.bx.y2);
        sprintf(buff, "%.6f %.6f %.6f %.6f %.6f\n",
                det.score, det.offset.x1, det.offset.y1, det.offset.x2, det.offset.y2);
        fputs(buff, fp);
    }
    fclose(fp);
#endif

    // nms
    _nms(*dets, *n, 0.5, 1);
    int n_boxes = _count_nzero(*dets, *n);
    if (n_boxes == 0){
        *n = 0;
        *dets = realloc(*dets, 0);
        return;
    }

    // keep boxes
    int j = 0;
    detect* tmp = calloc(*n, sizeof(detect));
    memcpy(tmp, *dets, *n*sizeof(detect));          // store origin dets
    *dets = realloc(*dets, n_boxes*sizeof(detect)); // realloc dets

    for (i = 0; i < *n; i++ ){
        float score = tmp[i].score;
        if (score == 0) continue;
        (*dets)[j++] = tmp[i];
    }

    *n = n_boxes;
    free(tmp);

    // calculate landmark
    _cal_landmark(*dets, *n);

    // refine boxes
    _cal_box(*dets, *n);
}

// ================================================================================== // 

void detect_image(network *pnet, network *rnet, network *onet, image im, int* n, detect** dets, params p)
{
    _detect_pnet(pnet, im, n, dets, p);
	_detect_rnet(rnet, im, n, dets, p);
	_detect_onet(onet, im, n, dets, p);
}

void show_detect(image im, detect* dets, int n, char* winname, int pause, int showscore, int showbox, int showmark)
{
    image tmp = rgb_to_bgr(copy_image(im));
#if 0
    IplImage* ipl = image_to_ipl(tmp);
    CvFont font; cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 0.5, 0.5, 1, 2, 8);

    for (int i = 0; i < n; i++ ){
        detect det = dets[i];
        float score = det.score;
        bbox bx = det.bx;
        landmark mk = det.mk;

        if (showscore){
            char buff[256];
            sprintf(buff, "%.2f", score);
            cvPutText(ipl, buff, cvPoint((int)bx.x1, (int)bx.y1),
                      &font, cvScalar(0, 0, 255, 0));
        }

        if (showbox){
            cvRectangle(ipl, cvPoint((int)bx.x1, (int)bx.y1),
                         cvPoint((int)bx.x2, (int)bx.y2),
                         cvScalar(255, 255, 255, 0), 1, 8, 0);
        }

        if (showmark){
            cvCircle(ipl, cvPoint((int)mk.x1, (int)mk.y1),
                         1, cvScalar(255, 255, 255, 0), 1, 8, 0);
            cvCircle(ipl, cvPoint((int)mk.x2, (int)mk.y2),
                         1, cvScalar(255, 255, 255, 0), 1, 8, 0);
            cvCircle(ipl, cvPoint((int)mk.x3, (int)mk.y3),
                         1, cvScalar(255, 255, 255, 0), 1, 8, 0);
            cvCircle(ipl, cvPoint((int)mk.x4, (int)mk.y4),
                         1, cvScalar(255, 255, 255, 0), 1, 8, 0);
            cvCircle(ipl, cvPoint((int)mk.x5, (int)mk.y5),
                         1, cvScalar(255, 255, 255, 0), 1, 8, 0);
        }
    }
    
    show_ipl(ipl, winname, pause);
#else
    for (int i = 0; i < n; i++ ){
        detect det = dets[i];
        bbox bx = det.bx;
        draw_box_width(tmp, bx.x1, bx.y1, bx.x2, bx.y2, 3, 1, 0, 0);
    }
    show_image(tmp, winname, pause);
#endif
}

#endif
