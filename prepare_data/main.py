from prepare_data.gen_12net_data        import save_12net_data
from prepare_data.gen_hard_example      import gen_hard_example
from prepare_data.gen_landmark_aug      import generateData_aug
from prepare_data.gen_imglist           import save_imglist
from prepare_data.gen_darknet_lists     import save_imglist_darknet

def gen_data(data_dir, net):

    if net == "PNet":
        save_12net_data(data_dir)

    elif net in ["RNet", "ONet"]:
        gen_hard_example(data_dir, net)
    
    else:
        print("Net type error! ")
        return 

    generateData_aug(data_dir, net, argument=True)
    save_imglist(data_dir, net)
    save_imglist_darknet(data_dir, net, train=0.8)

def main():
    data_dir = "/home/louishsu/Work/Codes/MTCNN_Darknet/data"
    
    ### For PNet
    gen_data(data_dir, "PNet")

    ### For RNet
    gen_data(data_dir, "RNet")

    ### For Onet
    gen_data(data_dir, "ONet")

if __name__ == "__main__":
    main()