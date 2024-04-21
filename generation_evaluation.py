'''
This code is used to evaluate the FID score of the generated images.
You should at least guarantee this code can run without any error on test set.
And whether this code can run is the most important factor for grading.
We provide the remaining code,  you can't modify the remaining code, all you should do are:
1. Modify the sample function to get the generated images from the model and ensure the generated images are saved to the gen_data_dir(line 12-18)
2. Modify how you call your sample function(line 31)
'''
'''
This code is used to evaluate the FID score of the generated images.
You should at least guarantee this code can run without any error on test set.
And whether this code can run is the most important factor for grading.
We provide the remaining code,  you can't modify the remaining code, all you should do are:
1. Modify the sample function to get the generated images from the model and ensure the generated images are saved to the gen_data_dir(line 12-18)
2. Modify how you call your sample function(line 31)
'''
from pytorch_fid.fid_score import calculate_fid_given_paths
from utils import *
from model import * 
from dataset import *
import os
import torch
# You should modify this sample function to get the generated images from your model
# This function should save the generated images to the gen_data_dir, which is fixed as 'samples'
# Begin of your code
    
sample_op = lambda x : sample_from_discretized_mix_logistic(x, 10)
def ta_les_crampte(model, gen_data_dir, sample_batch_size = 100, obs = (3,32,32), sample_op = sample_op):
    #generate images for each label, each label has 45 images (the conditionnalisation is integrated in the model)
    sample_t = sample(model, sample_batch_size, obs, sample_op)
    sample_t = rescaling_inv(sample_t)


    # Further splitting the remaining tensor into three parts of 25 each
    images_class_0, images_class_1, images_class_2, images_class_3 = sample_t.split(25) 

    for i in range (4):
        ###shamefull way to do it, I'm sorry :( ### 
        if i == 0: 
            save_images(images_class_0, os.path.join(gen_data_dir), i)
        elif i == 1: 
            save_images(images_class_1, os.path.join(gen_data_dir), i)
        elif i == 2: 
            save_images(images_class_2, os.path.join(gen_data_dir), i)
        elif i == 3: 
            save_images(images_class_3, os.path.join(gen_data_dir), i)
    

# End of your code

if __name__ == "__main__":
    ref_data_dir = "data/test"
    gen_data_dir = "samples"
    BATCH_SIZE=128
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not os.path.exists(gen_data_dir):
        os.makedirs(gen_data_dir)
    #Begin of your code
    #Load your model and generate images in the gen_data_dir
    model = PixelCNN(nr_resnet=5, nr_filters=160, nr_logistic_mix=10, input_channels=3, num_classes=4, label_embedding_dim=32)

    model = model.to(device)
    model.load_state_dict(torch.load('final_check.pth'))
    ta_les_crampte(model=model, gen_data_dir=gen_data_dir)
    #End of your code
    paths = [gen_data_dir, ref_data_dir]
    print("#generated images: {:d}, #reference images: {:d}".format(
        len(os.listdir(gen_data_dir)), len(os.listdir(ref_data_dir))))
    try:
        #changed batch size
        fid_score = calculate_fid_given_paths(paths, BATCH_SIZE, device, dims=192)
        print("Dimension {:d} works! fid score: {}".format(192, fid_score, gen_data_dir))
    except:
        print("Dimension {:d} fails!".format(192))
        
    print("Average fid score: {}".format(fid_score))
