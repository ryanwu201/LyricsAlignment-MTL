import argparse
import torch
from data import JamendoLyricsDataset
from model import AcousticModel
import utils, test

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="config/eval", config_name="csd_eval")
def main(cfg: DictConfig):
    args = cfg

    if args.model == "baseline":
        n_class = args.n_phone_class
    elif args.model == "MTL":
        n_class = (args.n_phone_class, args.n_pitch_class)
    else:
        ValueError("Invalid model type.")

    hparams = {
        "n_cnn_layers": 1,
        "n_rnn_layers": 3,
        "rnn_dim": args.rnn_dim,
        "n_class": n_class,
        "n_feats": 32,
        "stride": 1,
        "dropout": 0.1
    }

    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'

    model = AcousticModel(
        hparams['n_cnn_layers'], hparams['rnn_dim'], hparams['n_class'], \
        hparams['n_feats'], hparams['stride'], hparams['dropout']
    ).to(device)

    if 'cuda' in device:
        print("move model to gpu")
        model = utils.DataParallel(model)
        model.cuda()

    print('parameter count: ', str(sum(p.numel() for p in model.parameters())))

    print("Loading full model from checkpoint " + str(args.load_model))

    state = utils.load_model(model, args.load_model, args.cuda)

    test_data = JamendoLyricsDataset(args.sr, args.hdf_dir, args.dataset, args.jamendo_dir, args.sepa_dir,
                                     unit=args.unit, phones=args.phones, lang=args.lang)

    if 'predict_phoneme' not in args.keys() or ('predict_align' not in args.keys()):
        results_align = test.predict_align(args, model, test_data, device, args.model)
    else:
        if args.predict_align:
            results_align = test.predict_align(args, model, test_data, device, args.model)
        if args.predict_phoneme:
            results_phoneme = test.predict_phoneme(args, model, test_data, device, args.model)

    return


if __name__ == '__main__':
    main()
