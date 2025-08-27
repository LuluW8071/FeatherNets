import comet_ml
import os 
import argparse
import pytorch_lightning as pl 
import torch

# Load API
from dotenv import load_dotenv
load_dotenv()


from torch import optim
from torchmetrics.classification import (
    BinaryPrecision,
    BinaryRecall,
    BinaryAUROC,
    BinaryConfusionMatrix
)
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import CometLogger

from dataloader import CelebASpoofDataModule
from model import FeatherNet
from loss import FocalLoss


class FeatherNetTrainer(pl.LightningModule):
    def __init__(self, model, args):
        super(FeatherNetTrainer, self).__init__()
        self.model = model
        self.args = args
        
        self.loss_fn = FocalLoss(class_num=2, alpha=0.25, gamma=args.focal_gamma, size_average=True, device=args.device)

        # Setup Metrics
        self.precision = BinaryPrecision()
        self.recall = BinaryRecall()
        self.auroc = BinaryAUROC()
        self.bin_confmat = BinaryConfusionMatrix()
        
        self.save_hyperparameters(ignore=["model"])
        
        self.sync_dist = True if args.gpu_nodes > 1 else False

    def forward(self, x):
        return self.model(x)
        
    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(), 
            lr=self.args.learning_rate,
            momentum=0.9, 
            weight_decay=1e-4,
        )

        scheduler = {
            'scheduler': optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=10,     # restart every 10 epochs
                T_mult=2,   # increase cycle length each restart
                eta_min=1e-5
            ),
            'interval': 'epoch'
        }

        return [optimizer], [scheduler]
    
    def _common_step(self, X, y):
        outputs = self.forward(X)  # [batch, 1024]
        loss = self.loss_fn(outputs, y.long())
        
        return outputs, loss

    def training_step(self, batch, batch_idx):
        X, y = batch
        _, loss = self._common_step(X, y)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=self.sync_dist,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        outputs, loss = self._common_step(X, y)
        outputs = torch.softmax(outputs, dim=-1)
        soft_output = torch.softmax(outputs, dim=-1)
        _, predicted = torch.max(soft_output.data, 1)
        precision, recall, auc, apcer, bpcer, acer = self._compute_metrics(predicted, y)

        # Log all metrics using log_dict
        metrics = {
            'val_loss': loss,
            'val_precision': precision,
            'val_recall': recall,
            'val_auc': auc,
            'val_apcer': apcer,
            'val_bpcer': bpcer,
            'val_acer': acer
        }

        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.args.batch_size,
            sync_dist=self.sync_dist,
        )
        return {'val_loss': loss}
    
    def _compute_metrics(self, soft_output, y):
        """
        soft_output: [batch, 2] probabilities after softmax
        y: true labels [batch]
        """
        y_true = y.int()
        y_pred = torch.argmax(soft_output, dim=1)  # class predictions

        # Precision, recall, AUROC
        precision = self.precision(y_pred, y_true)
        recall = self.recall(y_pred, y_true)
        auc = self.auroc(soft_output[:,1], y_true)  # probability of positive class

        # ACER calculation
        real_mask = (y_true == 1)
        spoof_mask = (y_true == 0)

        tp = (y_pred[real_mask] == 1).sum().float()
        fn = (y_pred[real_mask] == 0).sum().float()

        tn = (y_pred[spoof_mask] == 0).sum().float()
        fp = (y_pred[spoof_mask] == 1).sum().float()

        apcer = fp / (fp + tn + 1e-8)
        bpcer = fn / (fn + tp + 1e-8)
        acer = (apcer + bpcer) / 2

        return precision, recall, auc, apcer, bpcer, acer
    
    
def main(args):
    comet_logger = CometLogger(api_key=os.getenv('API_KEY'), 
                               project=os.getenv('PROJECT_NAME'))
    
    dataloader = CelebASpoofDataModule(data_dir=args.data_path,
                                      label_dir=args.label_path,
                                      batch_size=args.batch_size, 
                                      num_workers=args.data_workers)
    
    # Call setup to initialize datasets
    dataloader.setup('fit')  

    # Initialize the model
    model = FeatherNet(se = True, avgdown=True).to(args.device)
    spoof_trainer = FeatherNetTrainer(model=model, args=args) 

    checkpoint_callback = ModelCheckpoint(
        monitor='val_acer',
        dirpath="./saved_checkpoint/",       
        filename='model-{epoch:02d}-{val_loss:.3f}-{val_acer:.4f}',                                             
        save_top_k=8,
        mode='min'
    )

    # Trainer Parameters
    trainer_args = {
        'accelerator': args.device,                                     # Device to use for training
        'devices': args.gpu_nodes,                                           # Number of GPUs to use for training
        'max_epochs': args.epochs,                                      # Maxm. no. of epochs to run                               
        'precision': args.precision,                                    # Precision to use for training
        'check_val_every_n_epoch': 1,                                   # No. of epochs to run validation
        'callbacks': [LearningRateMonitor(logging_interval='epoch'),    # Callbacks to use for training
                      EarlyStopping(monitor="val_acer", patience=5),
                      checkpoint_callback],
        'logger': comet_logger,                                         # Logger to use for training
    }

    if args.gpu_nodes > 1:
        trainer_args['strategy'] = args.dist_backend

    trainer = pl.Trainer(**trainer_args)

    # Fit the model to the training data using the Trainer's fit method.
    trainer.fit(spoof_trainer, dataloader)
    trainer.validate(spoof_trainer, dataloader)


if __name__  == "__main__":
    parser = argparse.ArgumentParser(description="Train")

    # Train Device Hyperparameters
    parser.add_argument('-d', '--device', default='cuda', type=str, choices=['cpu', 'cuda', 'mps'],
                        help='device to use for training, default cuda')
    parser.add_argument('-g', '--gpu_nodes', default=1, type=int, help='number of gpus per node')
    parser.add_argument('-w', '--data_workers', default=8, type=int,
                        help='n data loading workers, default 8')
    parser.add_argument('-db', '--dist_backend', default='ddp', type=str, help='which distributed backend to use for aggregating multi-gpu train')

    # Train and Test Directory Params
    parser.add_argument('--data_path', 
                        default="/teamspace/studios/this_studio/.cache/kagglehub/datasets/attentionlayer241/celeba-spoof-for-face-antispoofing/versions/2/CelebA_Spoof_/CelebA_Spoof/Data",
                        required=False, type=str, 
                        help='Folder path to load data')
    parser.add_argument('--label_path',
                        default="/teamspace/studios/this_studio/.cache/kagglehub/datasets/attentionlayer241/celeba-spoof-for-face-antispoofing/versions/2/CelebA_Spoof_/CelebA_Spoof/metas/intra_test", 
                        required=False, type=str,
                        help='Folder path to load label data')

    
    # General Train Hyperparameters
    parser.add_argument('--epochs', default=30, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=64, type=int, help='size of batch')
    parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--focal_gamma', default=2.0, type=float, help='gamma value for focal loss')
    parser.add_argument('--precision', default='16-mixed', type=str, help='precision')
    
    args = parser.parse_args()
    main(args)