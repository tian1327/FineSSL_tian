import os
import random
import argparse
import numpy as np
import torch

from utils.config import _C as cfg
from utils.logger import setup_logger
import open_clip

# from trainer import Trainer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    if cfg.KD:
        from trainer_kd import Trainer
    else:
        from trainer import Trainer

    if cfg.seed is not None:
        seed = cfg.seed
        print("Setting fixed seed: {}".format(seed))
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if cfg.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    # imbl = cfg.DATA.IMB_L
    # imbu = cfg.DATA.IMB_U
    # numl = cfg.DATA.NUM_L
    if cfg.output_dir is None:
        cfg.output_dir = os.path.join("./output", os.path.basename(args.cfg).rstrip(".yaml"))
    # else:
    #     cfg.output_dir = os.path.join(cfg.output_dir, cfg.DATA.NAME, f"NUML{numl}_imbl{imbl}_imbu{imbu}")

    print("** Config **")
    print(cfg)

    # New logic for open_clip
    backbone_key = cfg.backbone.split("_")[0]  # e.g. vitb32
    pretrain_key = pretrain_key = "_".join(cfg.backbone.split("_")[2:])


    model_name_map = {
        "vitb32": "ViT-B-32",
        "vitb16": "ViT-B-16",
        "vitl14": "ViT-L-14"
    }
    model_name = model_name_map[backbone_key]

    clip_model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrain_key)
    tokenizer = open_clip.get_tokenizer(model_name)

    os.makedirs(cfg.output_dir, exist_ok=True)
    trainer = Trainer(cfg, clip_model=clip_model, tokenizer=tokenizer)
    # End of new logic

    if cfg.eval_only:
        cfg.model_dir = cfg.model_dir if cfg.model_dir is not None else cfg.output_dir
        cfg.load_epoch = cfg.load_epoch if cfg.load_epoch is not None else cfg.num_epochs
        trainer.load_model(cfg.model_dir, epoch=cfg.load_epoch)
        trainer.test()
        return

    setup_logger(cfg.output_dir)
    # trainer = Trainer(cfg)

    if args.dry_run:
        trainer.build_data_loader()
        trainer.build_model()
        print("Dry run passed.")
        return

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true", help="Sanity check everything without training")
    parser.add_argument("--cfg", type=str, default="", help="path to config file")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="modify config options using the command-line")
    args = parser.parse_args()
    main(args)
