#include <opencv/cv.h>
#include "mtcnn.h"

#if 1

void run_mtcnn(int argc, char **argv)
{
    int help = find_arg(argc, argv, "--help");
    if(help || argc < 2){
        fprintf(stderr, "Usage:\n");
        fprintf(stderr, "    ./mtcnn <function>\n");
        fprintf(stderr, "\n");
        fprintf(stderr, "Optional:\n");
        fprintf(stderr, "    -v         video mode,     default `0`, image mode;\n");
        fprintf(stderr, "    --path     file path,      default `../images/test.*`;\n");
        fprintf(stderr, "    --index    camera index,   default `0`;\n");
        fprintf(stderr, "    -p         thresh for PNet,default `0.8`;\n");
        fprintf(stderr, "    -r         thresh for RNet,defalut `0.8`;\n");
        fprintf(stderr, "    -o         thresh for ONet,defalut `0.8`;\n");
        fprintf(stderr, "    --minface  minimal face,   default `96.0`;\n");
        fprintf(stderr, "    --scale    resize factor,  default `0.79`;\n");
        fprintf(stderr, "    --stride                   default `2`;\n");
        fprintf(stderr, "    --cellsize                 default `12`;\n");
        fprintf(stderr, "\n");
        fprintf(stderr, "Example:\n");
        fprintf(stderr, "    ./mtcnn -v 0\n");
        fprintf(stderr, "    ./mtcnn -v 0 --path [image path]\n");
        fprintf(stderr, "    ./mtcnn -v 1 --index 0\n");
        fprintf(stderr, "    ./mtcnn -v 1 --path [video path]\n");
        fprintf(stderr, "\n");
        return;
    }

    int mode = find_int_arg(argc, argv, "-v", 0);
    if (mode == 0){ // Image mode
        run_image(argc, argv);
    } else {        // Video mode
        run_video(argc, argv);
    }
}

int main(int argc, char **argv)
{
    run_mtcnn(argc, argv);
    return 0;
}

#elif 0

int main(int argc, char **argv)
{
    params p = fixedParams();

	network *pnet = load_mtcnn_net("PNet");
	network *rnet = load_mtcnn_net("RNet");
	network *onet = load_mtcnn_net("ONet");

    image im = load_image_color("../images/test.jpg", 0, 0);
    im = rgb_to_bgr(im);
    show_im(im, "image", 10);

    int n = 0;
    detect* dets = calloc(0, sizeof(detect));

    _detect_pnet(pnet, im, &n, &dets, p);
    show_detect(im, dets, n, "pnet", 0, 0, 1, 0);

	_detect_rnet(rnet, im, &n, &dets, p);
    show_detect(im, dets, n, "rnet", 0, 0, 1, 0);

	_detect_onet(onet, im, &n, &dets, p);
    show_detect(im, dets, n, "onet", 0, 0, 1, 1);

    free_image(im);
    free(dets);
	return 0;
}


#elif 0

void test_24()
{
    network *net = load_mtcnn_net("RNet");

    char* imgname[256];
    FILE *fp = fopen("/home/louishsu/Desktop/patches/c_24.txt", "w");

    for (int i = 0; i < 184; i++ ){
        sprintf(imgname, "/home/louishsu/Desktop/patches/24/%d.jpg", i);
        image im = load_image_color(imgname, 0, 0);
        im = rgb_to_bgr(im);

        float* X = im.data;
        network_predict(net, X);
        layer l = net->layers[net->n-1];

        char* output[256];
        for (int j = 0; j < 16; j++ ){
            if (j == 15){
                sprintf(output, "\n");
            } else {
                sprintf(output, "%.8f ", l.output[j]);
            }
            fputs(output, fp);
        }

        free_image(im);
    }

    fclose(fp);
}

void test_48()
{
    network *net = load_mtcnn_net("ONet");

    char* imgname[256];
    FILE *fp = fopen("/home/louishsu/Desktop/patches/c_48.txt", "w");

    for (int i = 0; i < 1; i++ ){
        sprintf(imgname, "/home/louishsu/Desktop/patches/48/%d.jpg", i);
        image im = load_image_color(imgname, 0, 0);
        im = rgb_to_bgr(im);

        float* X = im.data;
        network_predict(net, X);
        layer l = net->layers[net->n-1];

        char* output[256];
        for (int j = 0; j < 16; j++ ){
            if (j == 15){
                sprintf(output, "\n");
            } else {
                sprintf(output, "%.8f ", l.output[j]);
            }
            fputs(output, fp);
        }

        free_image(im);
    }

    fclose(fp);
}

int main(int argc, char **argv)
{
    test_24();
    test_48();
    return 0;
}

#elif 0
#include "parser.h"

int main(int argc, char **argv)
{
    // network *net = load_mtcnn_net("PNet");
    // network *net = load_network("cfg/PNet.cfg", "weights/PNet.weights", 0);
    network *net = parse_network_cfg("cfg/PNet.cfg");

    fprintf(stderr, "number of layers: %d\n", net->n);
    for (int i = 0; i < net->n; i++ ){
        layer l = net->layers[i];
        fprintf(stderr, "layer %d: [%size] size: %d | batch: %d w: %d h: %d c: %d -> w_out: %d h_out: %d c_out: %d \n",
                i, layer_type_to_string(l.type), sizeof(l),  
                l.batch, l.w, l.h, l.c, l.out_w, l.out_h, l.out_c);

    }

    return  0;
}

#endif





// int main()
// {
//     CvCapture *pCapture=NULL;
//     pCapture=cvCreateFileCapture("50254000.avi");
//     IplImage *pFrame=NULL;
//     cvNamedWindow("sor",0);
//     while (pFrame = cvQueryFrame(pCapture))
//     {
//         cvShowImage("sor",pFrame);
// 	cvWaitKey(10);
//     }
//     return 0;
// }