import time
import os
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import wandb
from utils import *
from model import * 
from dataset import *
from tqdm import tqdm
from pprint import pprint
import argparse
from pytorch_fid.fid_score import calculate_fid_given_paths
### added for computing acc every saved step ###
NUM_CLASSES = len(my_bidict)


def train_or_test(model, data_loader, optimizer, loss_op, device, args, epoch, mode):

    if mode == 'training':
        model.train()

    else:
        model.eval()

    deno = args.batch_size * np.prod(args.obs) * np.log(2.)
    loss_tracker = mean_tracker()
        

    for batch_idx, item in enumerate(tqdm(data_loader)):

        ###debugging ###
        if args.trainin_amount is not None and batch_idx >= args.trainin_amount:
            break
        ###


        model_input, label = item ### we also take the labels as inputs###
        model_input = model_input.to(device)

        ###sending to device ###
        ###we need intergers, not strings###

        ####changed here mode (rmv label & mode)####
        if mode == 'classify':
            logits = model(model_input)
        else : 
            class_strings = ['Class0', 'Class1', 'Class2', 'Class3']

            # Convert each string to the corresponding integer
            for i in range (len(label)):
                if label[i] == 'Class0':
                    label[i] = 0
                elif label[i] == 'Class1':
                    label[i] = 1
                elif  label[i] == 'Class2':
                    label[i] = 2
                elif label[i] == 'Class3':
                    label[i] = 3

            label = torch.tensor(label)
            label = label.to(device)



            ###we add the logits !
            model_output, logits  = model(model_input, 'train', label)


        ###adding classifciation loss as cross_entropy

        if mode != "classify": 
            classification_loss = F.cross_entropy(logits, label)

            loss = loss_op(model_input, model_output)

            ###adding the 2 loss 
            loss = loss + classification_loss

        loss_tracker.update(loss.item()/deno)
        if mode == 'training':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        
    if args.en_wandb:
        wandb.log({mode + "-Average-BPD" : loss_tracker.get_mean()})
        wandb.log({mode + "-epoch": epoch})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ###for debugging purposes : only train on a little amount of data) 
    parser.add_argument('-ld', '--trainin_amount', type=bool, default=None,
                            help='The pourcent of data used to train')
    ###
    parser.add_argument('-w', '--en_wandb', type=bool, default=False,
                            help='Enable wandb logging')
    parser.add_argument('-t', '--tag', type=str, default='default',
                            help='Tag for this run')
    
    # sampling
    parser.add_argument('-c', '--sampling_interval', type=int, default=5,
                        help='sampling interval')
    # data I/O
    parser.add_argument('-i', '--data_dir', type=str,
                        default='data', help='Location for the dataset')
    parser.add_argument('-o', '--save_dir', type=str, default='models',
                        help='Location for parameter checkpoints and samples')
    parser.add_argument('-sd', '--sample_dir',  type=str, default='samples',
                        help='Location for saving samples')
    parser.add_argument('-d', '--dataset', type=str,
                        default='cpen455', help='Can be either cifar|mnist|cpen455')
    parser.add_argument('-st', '--save_interval', type=int, default=10,
                        help='Every how many epochs to write checkpoint/samples?')
    parser.add_argument('-r', '--load_params', type=str, default=None,
                        help='Restore training from previous model checkpoint?')
    parser.add_argument('--obs', type=tuple, default=(3, 32, 32),
                        help='Observation shape')
    
    # model
    parser.add_argument('-q', '--nr_resnet', type=int, default=5,
                        help='Number of residual blocks per stage of the model')
    parser.add_argument('-n', '--nr_filters', type=int, default=160,
                        help='Number of filters to use across the model. Higher = larger model.')
    parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10,
                        help='Number of logistic components in the mixture. Higher = more flexible model')
    parser.add_argument('-l', '--lr', type=float,
                        default=0.0002, help='Base learning rate')
    parser.add_argument('-e', '--lr_decay', type=float, default=0.999995,
                        help='Learning rate decay, applied every step of the optimization')
    parser.add_argument('-b', '--batch_size', type=int, default=64,
                        help='Batch size during training per GPU')
    parser.add_argument('-sb', '--sample_batch_size', type=int, default=32,
                        help='Batch size during sampling per GPU')
    parser.add_argument('-x', '--max_epochs', type=int,
                        default=5000, help='How many epochs to run in total?')
    parser.add_argument('-s', '--seed', type=int, default=1,
                        help='Random seed to use')
    
    args = parser.parse_args()
    pprint(args.__dict__)
    check_dir_and_create(args.save_dir)
    
    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    model_name = 'pcnn_' + args.dataset + "_"
    model_path = args.save_dir + '/'
    if args.load_params is not None:
        model_name = model_name + 'load_model'
        model_path = model_path + model_name + '/'
    else:
        model_name = model_name + 'from_scratch'
        model_path = model_path + model_name + '/'
    
    job_name = "PCNN_Training_" + "dataset:" + args.dataset + "_" + args.tag
    
    if args.en_wandb:
        # start a new wandb run to track this script
        wandb.init(
            # set entity to specify your username or team name
            # entity="qihangz-work",
            # set the wandb project where this run will be logged
            project="CPEN455HW",
            # group=Group Name
            name=job_name,
        )
        wandb.config.current_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        wandb.config.update(args)

    #set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers':1, 'pin_memory':True, 'drop_last':True}

    # set data
    if "mnist" in args.dataset:
        ds_transforms = transforms.Compose([transforms.Resize((32, 32)), rescaling])
        train_loader = torch.utils.data.DataLoader(datasets.MNIST(args.data_dir, download=True, 
                            train=True, transform=ds_transforms), batch_size=args.batch_size, 
                                shuffle=True, **kwargs)
        
        test_loader  = torch.utils.data.DataLoader(datasets.MNIST(args.data_dir, train=False, 
                        transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
    
    elif "cifar" in args.dataset:
        ds_transforms = transforms.Compose([transforms.ToTensor(), rescaling])
        if args.dataset == "cifar10":
            train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=True, 
                download=True, transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
            
            test_loader  = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=False, 
                        transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
        elif args.dataset == "cifar100":
            train_loader = torch.utils.data.DataLoader(datasets.CIFAR100(args.data_dir, train=True, 
                download=True, transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
            
            test_loader  = torch.utils.data.DataLoader(datasets.CIFAR100(args.data_dir, train=False, 
                        transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
        else:
            raise Exception('{} dataset not in {cifar10, cifar100}'.format(args.dataset))
    
    elif "cpen455" in args.dataset:
        ds_transforms = transforms.Compose([transforms.Resize((32, 32)), rescaling])
        train_loader = torch.utils.data.DataLoader(CPEN455Dataset(root_dir=args.data_dir, 
                                                                  mode = 'train', 
                                                                  transform=ds_transforms), 
                                                   batch_size=args.batch_size, 
                                                   shuffle=True, 
                                                   **kwargs)
        # test_loader  = torch.utils.data.DataLoader(CPEN455Dataset(root_dir=args.data_dir, 
        #                                                           mode = 'test', 
        #                                                           transform=ds_transforms), 
        #                                            batch_size=args.batch_size, 
        #                                            shuffle=True, 
        #                                            **kwargs)
        val_loader  = torch.utils.data.DataLoader(CPEN455Dataset(root_dir=args.data_dir, 
                                                                  mode = 'validation', 
                                                                  transform=ds_transforms), 
                                                   batch_size=args.batch_size, 
                                                   shuffle=True, 
                                                   **kwargs)
    else:
        raise Exception('{} dataset not in {mnist, cifar, cpen455}'.format(args.dataset))
    
    args.obs = (3, 32, 32) # 3 channels, of size 32 x 32 
    input_channels = args.obs[0]
    
    loss_op   = lambda real, fake : discretized_mix_logistic_loss(real, fake)
    sample_op = lambda x : sample_from_discretized_mix_logistic(x, args.nr_logistic_mix)

    model = PixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters, 
                 nr_logistic_mix=args.nr_logistic_mix, input_channels=input_channels)

    model = model.to(device)

    if args.load_params:
        ### TO CHANGE, BECAUSE BUG ###
        model.load_state_dict(torch.load('/content/drive/MyDrive/CPEN455HW-2023W2/models/pcnn_cpen455_load_model_9.pth'))
        print('model parameters loaded')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)
    
    for epoch in tqdm(range(args.max_epochs)):
        train_or_test(model = model, 
                      data_loader = train_loader, 
                      optimizer = optimizer, 
                      loss_op = loss_op, 
                      device = device, 
                      args = args, 
                      epoch = epoch, 
                      mode = 'training')
        
        # decrease learning rate
        scheduler.step()
        # train_or_test(model = model,
        #               data_loader = test_loader,
        #               optimizer = optimizer,
        #               loss_op = loss_op,
        #               device = device,
        #               args = args,
        #               epoch = epoch,
        #               mode = 'test')
        
        train_or_test(model = model,
                      data_loader = val_loader,
                      optimizer = optimizer,
                      loss_op = loss_op,
                      device = device,
                      args = args,
                      epoch = epoch,
                      mode = 'val')
        
        ###changed so we don't need to sample at the first epoch
        if epoch % args.sampling_interval == 0 and epoch != 0:            
            print('......sampling......')


            #### this allows us to plot the 4 different classes separatly in wandb 
            sample_t = sample(model, args.sample_batch_size, args.obs, sample_op)
            sample_t = rescaling_inv(sample_t)

            number_images_each_class = args.sample_batch_size //4
            rest_in_class_0 = args.sample_batch_size % 4

            ###we then create "artificial" labels
            labels = torch.cat([
                torch.zeros(number_images_each_class + rest_in_class_0, dtype=torch.long),
                torch.ones(number_images_each_class, dtype=torch.long),
                torch.full((number_images_each_class,), 2, dtype=torch.long),
                torch.full((number_images_each_class,), 3, dtype=torch.long)
            ])



            ###label needs to be changed 
            images_class_0 = sample_t[:(number_images_each_class+rest_in_class_0)]
            images_other_class = sample_t[(number_images_each_class+rest_in_class_0):]

            # Further splitting the remaining tensor into three parts of 25 each
            images_class_1, images_class_2, images_class_3 = images_other_class.split(number_images_each_class) 

            for i in range (4):
                ###shamefull way to do it, I'm sorry :( ### 
                if i == 0: 
                    save_images(images_class_0, args.sample_dir, i)
                    sample_result_0 = wandb.Image(images_class_0, caption="class {} at epoch {}".format(i, epoch))
                    if args.en_wandb:
                        wandb.log({"samples class 0": sample_result_0})
                elif i == 1: 
                    save_images(images_class_1, args.sample_dir, i)
                    sample_result_1 = wandb.Image(images_class_1, caption="class {} at epoch {}".format(i, epoch))
                    if args.en_wandb:
                        wandb.log({"samples class 1": sample_result_1})
                elif i == 2: 
                    save_images(images_class_2, args.sample_dir, i)
                    sample_result_2 = wandb.Image(images_class_2, caption="class {} at epoch {}".format(i, epoch))
                    if args.en_wandb:
                        wandb.log({"samples class 2": sample_result_2})
                elif i == 3: 
                    save_images(images_class_3, args.sample_dir, i)
                    sample_result_3 = wandb.Image(images_class_3, caption="class {} at epoch {}".format(i, epoch))
                    if args.en_wandb:
                        wandb.log({"samples class 3": sample_result_3})
            
            gen_data_dir = args.sample_dir
            ref_data_dir = args.data_dir +'/test'
            paths = [gen_data_dir, ref_data_dir]
            try:
                fid_score = calculate_fid_given_paths(paths, 32, device, dims=192)
                print("Dimension {:d} works! fid score: {}".format(192, fid_score))
            except:
                print("Dimension {:d} fails!".format(192))
                
            if args.en_wandb:
                wandb.log({"FID": fid_score})
        
        ###Weird saving dir 
        if (epoch + 1) % args.save_interval == 0:

            

            ####

            # Construct the file path where you want to save the model
            model_file_path = '/content/drive/MyDrive/CPEN455HW-2023W2/models/{}_{}.pth'.format(model_name, epoch)
            
            # Save the model's state_dict or the entire model
            torch.save(model.state_dict(), model_file_path) 


            # OR if you want to save the entire model (not just the state_dict)
            # torch.save(model, model_file_path)