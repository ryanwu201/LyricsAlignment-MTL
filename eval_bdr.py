import os, argparse
import torch
from data import JamendoLyricsDataset
from model import AcousticModel, BoundaryDetection
import utils, test

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="config/eval", config_name="eval_bdr")
def main(cfg: DictConfig):
    args = cfg
    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'

    # acoustic model
    if args.model == "baseline":
        n_class = args.n_phone_class
    elif args.model == "MTL":
        n_class = (args.n_phone_class, args.n_pitch_class)
    else:
        ValueError("Invalid model type.")

    ac_hparams = {
        "n_cnn_layers": 1,
        "rnn_dim": args.rnn_dim,
        "n_class": n_class,
        "n_feats": 32,
        "stride": 1,
        "dropout": 0.1
    }

    ac_model = AcousticModel(
        ac_hparams['n_cnn_layers'], ac_hparams['rnn_dim'], ac_hparams['n_class'], \
        ac_hparams['n_feats'], ac_hparams['stride'], ac_hparams['dropout']
    ).to(device)

    # boundary model: fixed
    bdr_hparams = {
        "n_cnn_layers": 1,
        "rnn_dim": 32,  # a smaller rnn dim than acoustic model
        "n_class": 1,  # binary classification
        "n_feats": 32,
        "stride": 1,
        "dropout": 0.1,
    }

    bdr_model = BoundaryDetection(
        bdr_hparams['n_cnn_layers'], bdr_hparams['rnn_dim'], bdr_hparams['n_class'],
        bdr_hparams['n_feats'], bdr_hparams['stride'], bdr_hparams['dropout']
    ).to(device)

    if 'cuda' in device:
        print("move model to gpu")
        ac_model = utils.DataParallel(ac_model)
        ac_model.cuda()
        bdr_model = utils.DataParallel(bdr_model)
        bdr_model.cuda()

    print('parameter count (acoustic model): ', str(sum(p.numel() for p in ac_model.parameters())))
    print('parameter count (boundary model): ', str(sum(p.numel() for p in bdr_model.parameters())))

    print("Loading full model from checkpoint " + str(args.ac_model))
    print("Loading full model from checkpoint " + str(args.bdr_model))

    ac_state = utils.load_model(ac_model, args.ac_model, args.cuda)
    bdr_state = utils.load_model(bdr_model, args.bdr_model, args.cuda)

    test_data = JamendoLyricsDataset(args.sr, args.hdf_dir, args.dataset, args.jamendo_dir, args.sepa_dir,
                                     unit=args.unit, phones=args.phones)

    # predict with boundary detection
    results = test.predict_w_bdr(args, ac_model, bdr_model, test_data, device,
                                 args.alpha, args.model)


if __name__ == '__main__':
    main()
