#include "mtcnn.h"

#ifdef LIST

// ================================================================================== // 
detect* _detect(node* n)
{
    return ((detect*)n->val);
}

float _score(node* n)
{
    return _detect(n)->score;
}

bbox _box(node* n)
{
    return _detect(n)->bx;
}

bbox _offset(node* n)
{
    return _detect(n)->offset;
}

landmark _landmark(node* n)
{
    return _detect(n)->mk;
}

// ================================================================================== // 

void _nms(list* dets, float thresh, int mode)
{
    int status = -1;
    int i = 0, j = 0;

    list_attr(dets, _score);
    list_bsort(dets, _score);   // 升序排序，分数最高的在最后
    list_attr(dets, _score);

    node* n = dets->back;       // 指向分数最高的节点
    while(n){
        i = dets->size;
        j = i - 1;

        bbox a = _box(n);       // 当前分数最高的检测框
        node* p = n->prev;      // 前一个节点
        while(p){
            node* prev = p->prev;
            bbox b = _box(p);   // 遍历除分数最高框外的其他框
            float iou = bbox_iou(a, b, mode);

            printf("n[%4d], score: %.2f, p[%4d], score: %.2f, iou: %.6f\n",
                    i, _score(n), j, _score(p), iou);

            if (iou > thresh){
                if(p->next)p->next->prev = p->prev;
                if(p->prev)p->prev->next = p->next;
                free_current_node(p);
                --dets->size;

                status = list_check(dets);
                if (status!=-1){
                    printf("error, index: %d\n", status);
                }
            }
            p = prev;
            --j;

        }
        n = n->prev;
    }
    
    status = list_check(dets);
    status = list_check(dets);
}

void _cal_box(list* dets)
{
    node* n = dets->front;
    while(n){
        detect* det = _detect(n);
        bbox a = det->bx;
        bbox b = det->offset;

        float w = a.x2 - a.x1 + 1;
        float h = a.y2 - a.y1 + 1;
        a.x1 += b.x1*w;
        a.y1 += b.y1*h;
        a.x2 += b.x2*w;
        a.y2 += b.y2*h;
        
        det->bx = a;
        n->val = (void*)det;
        n = n->next;
    }
}

void _cal_landmark(list* dets)
{
    node* n = dets->front;
    while(n){
        detect* det = _detect(n);
        bbox a = det->bx;
        landmark b = det->mk;

        float w = a.x2 - a.x1 + 1;
        float h = a.y2 - a.y1 + 1;
        b.x1 += b.x1*w + a.x1;
        b.y1 += b.y1*h + a.y1;
        b.x2 += b.x2*w + a.x1;
        b.y2 += b.y2*h + a.y1;
        b.x3 += b.x3*w + a.x1;
        b.y3 += b.y3*h + a.y1;
        b.x4 += b.x4*w + a.x1;
        b.y4 += b.y4*h + a.y1;
        b.x5 += b.x5*w + a.x1;
        b.y5 += b.y5*h + a.y1;

        det->mk = b;
        n->val = (void*)det;
        n = n->next;
    }
}

void _square(list* dets)
{
    node* n = dets->front;
    while(n){
        bbox b = _box(n);
        
        float w = b.x2 - b.x1 + 1;
        float h = b.y2 - b.y1 + 1;

        float maxsize = _max(w, h);
        b.x1 += w/2 - maxsize/2;
        b.y1 += h/2 - maxsize/2;
        b.x2  = b.x1 + maxsize - 1;
        b.y2  = b.y1 + maxsize - 1;

        ((detect*)n->val)->bx = b;
        n = n->next;
    }
}

float* _crop_patch(image im, list* dets, float netsize)
{
    int size = netsize*netsize*im.c;
    float *X = calloc(dets->size, sizeof(float)*size);
    int i = 0, j = 0, k = 0, l = 0;

    node* n = dets->front;
    while(n){

        bbox a = _box(n);

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

        // resize patch
        image sized = resize_image(patch, (int)netsize, (int)netsize);

        // copy patch
        memcpy(X + i*size, sized.data, sizeof(float)*size);

        free_image(patch);
        free_image(sized);

        n = n->next;
        i++;
    }
    
    return X;
}

// ================================================================================== // 

void _detect_pnet(network *net, image im, list* dets, params p)
{
    float NETSIZE = 12.;
    int i, j;

    float cur_scale = NETSIZE / p.min_face;
    image cur_img = resize_image_scale(im, cur_scale);

    while(_min(cur_img.h, cur_img.w) >= NETSIZE)
    {

        // forward network
        resize_network(net, cur_img.w, cur_img.h);
        network_predict(net, cur_img.data);
        layer l = net->layers[net->n-1];

        // generate bbox
        for (i = 0; i < l.h; i++ ){
            for (j = 0; j < l.w; j++ ){

                float score = logistic_activate(l.output[j + i*l.w]);
                if (score > p.pthresh){

                    detect* det = calloc(1, sizeof(detect));
                    
                    // boxes
                    det->score = score;
                    det->bx.x1 = j*p.stride / cur_scale;
                    det->bx.y1 = i*p.stride / cur_scale;
                    det->bx.x2 = (j*p.stride + p.cellsize) / cur_scale;
                    det->bx.y2 = (i*p.stride + p.cellsize) / cur_scale;

                    // offsets
                    det->offset.x1 = l.output[j + i*l.w + 1*l.w*l.h];
                    det->offset.y1 = l.output[j + i*l.w + 2*l.w*l.h];
                    det->offset.x2 = l.output[j + i*l.w + 3*l.w*l.h];
                    det->offset.y2 = l.output[j + i*l.w + 4*l.w*l.h];

                    list_insert(dets, det);
                }
            }
        }

        // update scale
        cur_scale *= p.scale;
        free_image(cur_img);
        cur_img = resize_image_scale(im, cur_scale);
    }
    free_image(cur_img);

    if (dets->size == 0)
        return;

    _nms(dets, 0.6, 0);
    _cal_box(dets);
}

void _detect_rnet(network *net, image im, list* dets, params p)
{
    if (dets->size == 0) return;

    float NETSIZE = 24.;
    set_batch_network(net, dets->size);
    resize_network(net, (int)NETSIZE, (int)NETSIZE);

    // crop data from origin image
    _square(dets);
    float* X = _crop_patch(im, dets, NETSIZE);

    // forward network
    network_predict(net, X); free(X);
    layer l = net->layers[net->n-1];

    // update dets
    int i = 0;
    node* n = dets->front;
    while(n){
        
        float *output = l.output + i*l.outputs;
        float score = logistic_activate(*output);

        if (score < p.rthresh){
            // delete current node
            if(n->next)n->next->prev = n->prev;
            if(n->prev)n->prev->next = n->next;
            node* tmp = n->next;
            free_current_node(n);
            --dets->size;
            n = tmp;
            continue;
        }

        detect* det = _detect(n);
        det->score = score;
        det->offset.x1 = *(output + 1);
        det->offset.y1 = *(output + 2);
        det->offset.x2 = *(output + 3);
        det->offset.y2 = *(output + 4);
        n->val = (void*)det;

        n = n->next;
        i++;
    }

    _nms(dets, 0.5, 0);
    _cal_box(dets);
}

void _detect_onet(network *net, image im, list* dets, params p)
{
    if (dets->size == 0) return;

    float NETSIZE = 48.;
    set_batch_network(net, dets->size);
    resize_network(net, (int)NETSIZE, (int)NETSIZE);

    // crop data from origin image
    _square(dets);
    float* X = _crop_patch(im, dets, NETSIZE);

    // forward network
    network_predict(net, X); free(X);
    layer l = net->layers[net->n-1];

    // update dets
    int i = 0;
    node* n = dets->front;
    while(n){
        
        float *output = l.output + i*l.outputs;
        float score = logistic_activate(*output);

        if (score < p.othresh){
            // delete current node
            if(n->next)n->next->prev = n->prev;
            if(n->prev)n->prev->next = n->next;
            node* tmp = n->next;
            free_current_node(n);
            --dets->size;
            n = tmp;
            continue;
        }

        detect* det = _detect(n);
        det->score = score;
        det->offset.x1 = *(output + 1);
        det->offset.y1 = *(output + 2);
        det->offset.x2 = *(output + 3);
        det->offset.y2 = *(output + 4);
        det->mk.x1 = *(output + 5);
        det->mk.y1 = *(output + 6);
        det->mk.x2 = *(output + 7);
        det->mk.y2 = *(output + 8);
        det->mk.x3 = *(output + 9);
        det->mk.y3 = *(output + 10);
        det->mk.x4 = *(output + 11);
        det->mk.y4 = *(output + 12);
        det->mk.x5 = *(output + 13);
        det->mk.y5 = *(output + 14);
        n->val = (void*)det;

        n = n->next;
        i++;
    }

    _nms(dets, 0.5, 1);
    _cal_landmark(dets);
    _cal_box(dets);
}

void detect_image(network *pnet, network *rnet, network *onet, image im, list* dets, params p)
{
    _detect_pnet(pnet, im, dets, p);
	_detect_rnet(rnet, im, dets, p);
	_detect_onet(onet, im, dets, p);
}
// ================================================================================== // 

void show_detect(image im, list* dets, char* winname, int pause, int showscore, int showbox, int showmark)
{
    image tmp = copy_image(im);
    IplImage* ipl = image_to_ipl(tmp);
    CvFont font; cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 0.5, 0.5, 1, 2, 8);

    if (dets->size != 0){
        node* n = dets->front;
        while(n){
            detect* det = _detect(n);
            float score = det->score;
            bbox bx = det->bx;
            landmark mk = det->mk;

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

            n = n->next;
        }
    }

    show_ipl(ipl, winname, pause);
}

void run_mtcnn(int argc, char **argv)
{
    if(argc < 2){
        fprintf(stderr, "usage: ./mtcnn [image path] \n");
        fprintf(stderr, "  Optional:\n");
        fprintf(stderr, "    -p         thresh for PNet\n");
        fprintf(stderr, "    -r         thresh for RNet\n");
        fprintf(stderr, "    -o         thresh for ONet\n");
        fprintf(stderr, "    --minface  minimal face, default 12.\n");
        fprintf(stderr, "    --scale    resize factor, default 0.79\n");
        fprintf(stderr, "    --stride   default 2\n");
        fprintf(stderr, "    --cellsize default 12\n");
        fprintf(stderr, "  no file found, use default image\n");
    }

    // load image
    char* imgpath = argc < 2? "../images/test.jpg": argv[1];
    image im = load_image_color(imgpath, 0, 0);
    im = rgb_to_bgr(im);
    show_im(im, "image", 10);

    params p = initParams(argc, argv);

    network *pnet = load_mtcnn_net("PNet");
    network *rnet = load_mtcnn_net("RNet");
    network *onet = load_mtcnn_net("ONet");

    list* dets = make_list();

#if 1
    double time = 0;

    time = what_time_is_it_now();
    _detect_pnet(pnet, im, dets, p);
    printf("PNet | Predicted in %f seconds.\n", what_time_is_it_now()-time);
    // image tmp = copy_image(im);
    image tmp = make_empty_image(im.h, im.w, im.c);
    tmp.data = calloc(tmp.h*tmp.w*tmp.c, sizeof(float));
    memcpy(tmp.data, im.data, tmp.h*tmp.w*tmp.c);
    free_image(tmp);
    show_detect(im, dets, "pnet", 0, 0, 1, 0);
    
    time = what_time_is_it_now();
    _detect_rnet(rnet, im, dets, p);
    printf("RNet | Predicted in %f seconds.\n", what_time_is_it_now()-time);
    show_detect(im, dets, "rnet", 0, 0, 1, 0);

    time = what_time_is_it_now();
    _detect_onet(onet, im, dets, p);
    printf("ONet | Predicted in %f seconds.\n", what_time_is_it_now()-time);
    show_detect(im, dets, "onet", 0, 0, 1, 1);

    free_image(im);
    free_list(dets);
    free_network(pnet);
    free_network(rnet);
    free_network(onet);
#else
    double time = what_time_is_it_now();
    detect_image(pnet, rnet, onet, im, dets, p);
    printf("Predicted in %f seconds.\n", what_time_is_it_now()-time);
    show_detect(im, dets, "mtcnn", 0, 0, 1, 1);
#endif

}

#endif