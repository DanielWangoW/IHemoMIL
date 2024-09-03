import os
import logging
import argparse


import torch

from ihemomil.model import backbone, pooling
from ihemomil.model.ihemomil_model import IHemoMILModel

from ihemomil.utils import set_logger, data_selecter, results_log, plot_conf_mat
from ihemomil.utils import BulidModel, BACKBONE_ZOO, POOLING_METHODS

# create logger 
logger = logging.getLogger("IHEMOMIL.Train")

'''
Get train configuration from command line with argparse
: param is_train: bool, train or test
: param dataset: str, dataset name 1. web_traffic 2. mp_ppg_TBME 3. mp_ppg_BIDMC 4. mp_ppg_VitalDB, 5. bpf, 6. mimic_af, 7. dalia
: param data_path: str, data path
: param checkpoint: str, checkpoint name
: param channels: int, number of channels, default 1
: param backbone: str, backbone model for featuer extraction ["fcn", "inceptiontime", "resnet", "mlp", "transformer"]
: param pooling: str, pooling method for feature aggregation ["gap", "rap", "ins", "atte", "ratte", "addi", "raddi", "conj", "rconj"]
: param d_model: int, dimension of model, default 128
: param dropout: float, dropout rate of projection matrix, default 0.1
: param p_rank: float, select p_rank% for projection matrix, default 0.2
: param p_alpha: float, select alpha% for projection matrix, default 0.1
: param d_attn: int, dimension of attention, default 8
: param apply_positional_encoding: bool, apply positional encoding or not, default True
: param batch_size: int, batch size, default 2048, min(dataset size // 5, 2048)
: param epochs: int, number of epochs, default 1500
: param learning_rate: float, learning rate, default 0.001
: param use_gpu: bool, use gpu or not, default True
: param gpu_id: int, gpu id, default 0
'''


# get configurations of the model
parser = argparse.ArgumentParser()

# data loader
parser.add_argument("--is_train", type=bool, default=True, help="train or test")
parser.add_argument("--dataset", type=str, default="simhf3k", help="dataset name")
parser.add_argument("--data_path", type=str, default="data", help="data path")
parser.add_argument("--checkpoint", type=str, default="checkpoint-temp", help="checkpoint name")
parser.add_argument("--channels", type=int, default=1, help="number of channels")

# basic model configurations
parser.add_argument("--backbone", type=str, default="inceptiontime", help="backbone model for featuer extraction")
parser.add_argument("--pooling", type=str, default="atte", help="pooling method for feature aggregation")

# model hyperparameters
parser.add_argument("--d_model", type=int, default=128, help="dimension of model")
parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate of projection matrix")
parser.add_argument("--p_rank", type=float, default=0.2, help="select p_rank% for projection matrix")
parser.add_argument("--p_alpha", type=float, default=0.1, help="select alpha% for projection matrix")
parser.add_argument("--d_attn", type=int, default=8, help="dimension of attention")
parser.add_argument("--apply_positional_encoding", type=bool, default=True, help="apply positional encoding or not")

# optimization
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")

# gpu settings
parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu or not")
parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")


if __name__ == '__main__':

    args = parser.parse_args()

    data_dir = os.path.join(args.data_path, args.dataset)
    model_name = args.backbone + "_" + args.pooling
    checkpoint_dir = os.path.join(args.checkpoint, args.dataset, model_name)
    # create checkpoint directory
    try:
        os.makedirs(checkpoint_dir)
    except FileExistsError:
        pass

    # set logger
    set_logger(os.path.join(checkpoint_dir, "train.log"))
    logger.info("IHEMOMIL: toward to Interpretable HEMOdynamic fluctuation in photoplethysmograph")
    
    # use GPU if acailabel
    if torch.cuda.is_available() and args.use_gpu:
        device = torch.device("cuda:{}".format(args.gpu_id))
        logger.info("Using GPU: {}".format(torch.cuda.get_device_name(device)))
    else:
        device = torch.device("cpu")
        logger.info("Using CPU...")
    
    # load data
    logger.info("Loading data...")
    train_dataset, test_dataset = data_selecter(args.dataset)
    n_classes = train_dataset.n_clz
    

    # build model
    logger.info("Building model...")
    backbone = BACKBONE_ZOO.get(args.backbone, None)
    pooling = POOLING_METHODS.get(args.pooling, None)
    logger.info("Backbone: {}".format(backbone))
    logger.info("Pooling: {}".format(pooling))

    if args.pooling == 'gap' or args.pooling == 'rap' or args.pooling == 'ins':
        net = BulidModel(
            backbone(args.channels),
            pooling(args.d_model, n_classes, args.dropout, args.p_rank, args.apply_positional_encoding))
    
    elif args.pooling == 'atte' or args.pooling == 'addi' or args.pooling == 'conj': 
        net = BulidModel(
            backbone(args.channels),
            pooling(args.d_model, n_classes, args.d_attn, args.dropout, args.apply_positional_encoding))
    else:
        net = BulidModel(
            backbone(args.channels),
            pooling(args.d_model, n_classes, args.d_attn, args.dropout, 
                    args.p_alpha, args.apply_positional_encoding),
            )
    logger.info("d_model: {}, d_dropout: {}, p_alpha: {}".format(args.d_model, args.dropout, args.p_alpha))
    
    ihemomil = IHemoMILModel(model_name, device, n_classes, net)

    logger.info("trainable parameters: {}".format(ihemomil.num_params()))

    if args.is_train:
        # train model
        logger.info("Training model...")
        ihemomil.fit(train_dataset, args.batch_size, args.epochs, args.learning_rate)
        # save model
        logger.info("Saving model...")
        save_model_path = os.path.join(checkpoint_dir, "model.pth")
        ihemomil.save_weights(save_model_path)
    else:
        # load model
        logger.info("Loading model...")
        load_model_path = os.path.join(checkpoint_dir, "model.pth")
        ihemomil.load_weights(load_model_path)

    # evaluate model
    logger.info("Evaluating model...")

    # Evaluate predictive performance on train and test splits
    train_results_dict = ihemomil.evaluate(train_dataset)
    test_results_dict = ihemomil.evaluate(test_dataset)

    # Evaluate interpretability on train and test splits
    train_aopcr, train_ndcg = ihemomil.evaluate_interpretability(train_dataset)
    test_aopcr, test_ndcg = ihemomil.evaluate_interpretability(test_dataset)

    header = ["Split", "Accuracy", "AUROC", "Loss", "AOPCR", "NDCG@n"]
    train_row = ["Train", train_results_dict["acc"], train_results_dict["auroc"],
                 train_results_dict["loss"], train_aopcr, train_ndcg,]
    test_row = ["Test", test_results_dict["acc"], test_results_dict["auroc"],
                test_results_dict["loss"], test_aopcr, test_ndcg,]
    results_data = [header, train_row, test_row]


    # Print results table
    results_log(results_data)
    # Plot confusion matrix
    plot_conf_mat(test_results_dict["conf_mat"], checkpoint_dir, args.dataset)

    logger.info("Done!\n")
