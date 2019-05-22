#include "mtcnn.h"

#define H 400
#define W 800

static int g_videoDone = 0;
static char* winname = "frame";
static CvFont font;

static CvCapture* g_cvCap = NULL;
static image g_imFrame[3];
static int g_index = 0;

static double g_time;
static double g_fps;
static int g_running = 0;
static detect* g_dets = NULL;
static int g_ndets = 0;

static params p;
static network* pnet;
static network* rnet;
static network* onet;

image _frame()
{
    IplImage* iplFrame = cvQueryFrame(g_cvCap);
    // IplImage* iplFrame = cvLoadImage("/home/louishsu/Desktop/test.jpg", CV_LOAD_IMAGE_ANYCOLOR);
    image dst = ipl_to_image(iplFrame);
    return dst;
}

void* read_frame_in_thread(void* ptr)
{
    free_image(g_imFrame[g_index]);
    g_imFrame[g_index] = _frame();
    if (g_imFrame[g_index].data == 0){
        g_videoDone = 1;
        return 0;
    }
    return 0;
}

void* detect_frame_in_thread(void* ptr)
{
    g_running = 1;

    image frame = g_imFrame[(g_index + 2) % 3];
    g_dets = realloc(g_dets, 0); g_ndets = 0;
    detect_image(pnet, rnet, onet, frame, &g_ndets, &g_dets, p);
#if 1
    for (int i = 0; i < g_ndets; i++ ){
        detect det = g_dets[i];
        bbox bx = det.bx;
        draw_box_width(frame, bx.x1, bx.y1, bx.x2, bx.y2, 3, 0, 0, 1);  // b, g, r
    }
#else
    IplImage* iplFrame = image_to_ipl(g_imFrame[(g_index + 2) % 3]);
    for (int i = 0; i < g_ndets; i++ ){
        detect det = g_dets[i];
        float score = det.score;
        bbox bx = det.bx;
        landmark mk = det.mk;

        char buff[256];
        sprintf(buff, "%.2f", score);
        cvPutText(iplFrame, buff, cvPoint((int)bx.x1, (int)bx.y1),
                    &font, cvScalar(0, 0, 255, 0));

        cvRectangle(iplFrame, cvPoint((int)bx.x1, (int)bx.y1),
                    cvPoint((int)bx.x2, (int)bx.y2),
                    cvScalar(255, 255, 255, 0), 1, 8, 0);

        cvCircle(iplFrame, cvPoint((int)mk.x1, (int)mk.y1),
                    1, cvScalar(255, 255, 255, 0), 1, 8, 0);
        cvCircle(iplFrame, cvPoint((int)mk.x2, (int)mk.y2),
                    1, cvScalar(255, 255, 255, 0), 1, 8, 0);
        cvCircle(iplFrame, cvPoint((int)mk.x3, (int)mk.y3),
                    1, cvScalar(255, 255, 255, 0), 1, 8, 0);
        cvCircle(iplFrame, cvPoint((int)mk.x4, (int)mk.y4),
                    1, cvScalar(255, 255, 255, 0), 1, 8, 0);
        cvCircle(iplFrame, cvPoint((int)mk.x5, (int)mk.y5),
                    1, cvScalar(255, 255, 255, 0), 1, 8, 0);
    }
    g_imFrame[(g_index + 2) % 3] = ipl_to_image(iplFrame);
#endif
    printf("\033[2J");
    printf("\033[1;1H");
    printf("\nFPS:%.1f\n", g_fps);
    printf("Objects:%d\n", g_ndets);
    g_running = 0;
}

void* display_frame_in_thread(void* ptr)
{
    int c = show_image(rgb_to_bgr(g_imFrame[(g_index + 1) % 3]), winname, 1);
    if (c != -1) c = c%256;
    if (c == 27) {
        g_videoDone = 1;
        return 0;
    }
    return 0;
}

void run_video(int argc, char **argv)
{
    pnet = load_mtcnn_net("PNet");
    rnet = load_mtcnn_net("RNet");
    onet = load_mtcnn_net("ONet");
    printf("\n\n");

    printf("Initializing Capture...");
    int index = find_int_arg(argc, argv, "--index", 0);
    if (index < 0){
        char* filepath = find_char_arg(argc, argv, "--path", "../images/test.mp4");
        if(0==strcmp(filepath, "../images/test.mp4")){
            fprintf(stderr, "Using default: %s\n", filepath);
        }
        g_cvCap = cvCaptureFromFile(filepath);
    } else {
        g_cvCap = cvCaptureFromCAM(index);
    }
    if (!g_cvCap){
        printf("failed!\n");
        return;
    }
    cvSetCaptureProperty(g_cvCap, CV_CAP_PROP_FRAME_HEIGHT, H);
    cvSetCaptureProperty(g_cvCap, CV_CAP_PROP_FRAME_WIDTH, W);
    
    g_imFrame[0] = _frame();
    g_imFrame[1] = copy_image(g_imFrame[0]);
    g_imFrame[2] = copy_image(g_imFrame[0]);

    cvNamedWindow(winname, CV_WINDOW_AUTOSIZE);
    cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 0.5, 0.5, 1, 2, 8);
    printf("OK!\n");

    printf("Initializing detection...");
    p = initParams(argc, argv);
    g_dets = calloc(0, sizeof(detect)); g_ndets = 0;
    printf("OK!\n");

    pthread_t thread_read;
    pthread_t thread_detect;

    g_time = what_time_is_it_now();
    while(!g_videoDone){
        g_index = (g_index + 1) % 3;

        if(pthread_create(&thread_read, 0, read_frame_in_thread, 0)) error("Thread read create failed");
        if(pthread_create(&thread_detect, 0, detect_frame_in_thread, 0)) error("Thread detect create failed");
        
        g_fps = 1./(what_time_is_it_now() - g_time);
        g_time = what_time_is_it_now();
        display_frame_in_thread(0);
        
        pthread_join(thread_read, 0);
        pthread_join(thread_detect, 0);
    }
    for (int i = 0; i < 3; i++ ){
        free_image(g_imFrame[i]);
    }
    free(g_dets);
    cvReleaseCapture(g_cvCap);
    cvDestroyWindow(winname);

    free_network(pnet);
    free_network(rnet);
    free_network(onet);
}
