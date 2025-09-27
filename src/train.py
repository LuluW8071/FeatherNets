import comet_ml
import os 
import argparse
import pytorch_lightning as pl 
import torch

# Load API
from dotenv import load_dotenv
load_dotenv()


from torch import optim 
from torch.nn import functional as F
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import CometLogger

from torchmetrics.classification import (
    Accuracy,
    Precision,
    Recall,
    F1Score,
    AUROC
)

from dataloader import SpoofDataModule
from feathernet.feathernet import FeatherNet
from multi_task_criterion import MultiTaskCriterion



class FeatherNetTrainer(pl.LightningModule):
    def __init__(self, model, args):
        super(FeatherNetTrainer, self).__init__()
        self.model = model
        self.args = args

        self.loss_fn = MultiTaskCriterion(alpha=0.7, gamma=args.focal_gamma, device=args.device) # [real_alpha, spoof_alpha]

        # Metrics
        self.accuracy = Accuracy(task="binary", num_classes=2)
        self.precision = Precision(task="binary", num_classes=2)
        self.recall = Recall(task="binary", num_classes=2)
        self.f1_score = F1Score(task="binary", num_classes=2)
        self.auroc = AUROC(task="binary")

        self.val_losses, self.test_losses = [], []
        self.val_preds, self.test_preds = [], []
        self.val_probs, self.test_probs = [], []
        self.val_labels, self.test_labels = [], []
        
        self.sync_dist = True if args.gpu_nodes > 1 else False

        self.save_hyperparameters(ignore=["model"])

        
    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(), 
            lr=self.args.learning_rate,
            momentum=0.9, 
            weight_decay=1e-4,
        )

        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.5,
                patience=1,
                threshold=3e-2,
                threshold_mode='rel',
                min_lr=1e-5
            ),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1 
        }

        return [optimizer], [scheduler]
    
    def _common_val_step(self, X, y, threshold: float = 0.5):
        outputs = self.model._forward_train(X)
        loss = self.loss_fn(outputs, y, eval=True)      # Only classification loss
        probs = F.softmax(outputs[0], dim=1)[:, 1]
        preds = (probs > threshold).long()
        return preds, probs, loss

    def training_step(self, batch, batch_idx):
        X, y = batch
        outputs = self.model._forward_train(X)
        spoof_loss, depth_loss, loss = self.loss_fn(outputs, y, eval=False)
        metrics = {
            'spoof_loss': spoof_loss,
            'depth_loss': depth_loss,
            'train_loss': loss,
        }
        self.log_dict(
            metrics,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=self.sync_dist,
        )
        return {'loss': loss, 'spoof_loss': spoof_loss, 'depth_loss': depth_loss}

    def validation_step(self, batch, batch_idx):
        X, y = batch
        preds, probs, val_loss = self._common_val_step(X, y)

        # Store batch outputs for epoch aggregation
        self.val_losses.append(val_loss)
        self.val_preds.append(preds)
        self.val_probs.append(probs)
        self.val_labels.append(y[0])
        return {'val_loss': val_loss}


    def on_validation_epoch_end(self):
        # Concatenate all batches
        all_preds = torch.cat(self.val_preds)
        all_probs = torch.cat(self.val_probs)
        all_y = torch.cat(self.val_labels)
        avg_loss = torch.stack(self.val_losses).mean()

        # Compute metrics using centralized function
        acc, precision, recall, f1_score, auc, acer = self._compute_metrics(all_preds, all_probs, all_y)

        # Log metrics
        prog_log_metrics = {'val_loss': avg_loss, 'val_acer': acer}
        metrics = {
            'val_acc': acc,
            'val_precision': precision,
            'val_recall': recall,
            'val_f1': f1_score,
            'val_auc': auc
        }

        self.log_dict(prog_log_metrics, on_step=False, on_epoch=True, prog_bar=True,
                    logger=True, batch_size=self.args.batch_size, sync_dist=self.sync_dist)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False,
                    logger=True, batch_size=self.args.batch_size, sync_dist=self.sync_dist)

        # Clear lists for next epoch
        self.val_losses.clear()
        self.val_preds.clear()
        self.val_probs.clear()
        self.val_labels.clear()


    def test_step(self, batch, batch_idx):
        X, y = batch
        preds, probs, test_loss = self._common_val_step(X, y)

        self.test_losses.append(test_loss)
        self.test_preds.append(preds)
        self.test_probs.append(probs)
        self.test_labels.append(y[0])
        return {'test_loss': test_loss}


    def on_test_epoch_end(self):
        all_preds = torch.cat(self.test_preds)
        all_probs = torch.cat(self.test_probs)
        all_y = torch.cat(self.test_labels)
        avg_loss = torch.stack(self.test_losses).mean()

        # Compute metrics centrally
        acc, precision, recall, f1_score, auc, acer = self._compute_metrics(all_preds, all_probs, all_y)

        metrics = {
            'test_loss': avg_loss,
            'test_acc': acc,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1': f1_score,
            'test_auc': auc,
            'test_acer': acer
        }

        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False,
                    logger=True, batch_size=self.args.batch_size, sync_dist=self.sync_dist)

        # Clear lists
        self.test_losses.clear()
        self.test_preds.clear()
        self.test_probs.clear()
        self.test_labels.clear()

    def _compute_metrics(self, y_pred: torch.Tensor, y_prob: torch.Tensor, y: torch.Tensor):
        """
        Compute ACER, APCER, BPCER using vectorized counting.
        
        Args:
            y_pred: predicted class labels [batch] (already argmaxed)
            y: true labels [batch]
        
        Returns:
            apcer, bpcer, acer
        """
        y_true = y.int()

        real_mask = (y_true == 0)
        spoof_mask = (y_true == 1)

        total_attack_error = (y_pred[spoof_mask] != y_true[spoof_mask]).sum().float()
        total_normal_error = (y_pred[real_mask] != y_true[real_mask]).sum().float()

        total_attack_samples = spoof_mask.sum().float()
        total_normal_samples = real_mask.sum().float()

        apcer = total_attack_error / (total_attack_samples + 1e-8)
        bpcer = total_normal_error / (total_normal_samples + 1e-8)
        acer = (apcer + bpcer) / 2

        acc = self.accuracy(y_pred, y_true)
        precision = self.precision(y_pred, y_true)
        recall = self.recall(y_pred, y_true)
        f1_score = self.f1_score(y_pred, y_true)
        auc = self.auroc(y_prob, y_true) 

        return acc, precision, recall, f1_score, auc, acer
    
def main(args):
    comet_logger = CometLogger(api_key=os.getenv('API_KEY'), 
                               project=os.getenv('PROJECT_NAME'))
    
    
    dataloader = SpoofDataModule(lcc_dir=args.lcc_dir,
                                batch_size=args.batch_size, 
                                num_workers=args.data_workers)
    
    # Call setup to initialize datasets
    dataloader.setup('fit')  
    
    # Initialize the model
    model = FeatherNet() 
    spoof_trainer = FeatherNetTrainer(model=model, args=args) 
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acer",
        dirpath=f"saved_checkpoint/{comet_logger.name}/version_{comet_logger.version}",
        filename="mod_feathernet-{epoch:02d}-{val_acer:.5f}",
        save_top_k=3,
        mode="min"
    )

    # Trainer Parameters
    trainer_args = {
        'accelerator': args.device,                                     # Device to use for training
        'devices': args.gpu_nodes,                                      # Number of GPUs to use for training
        'max_epochs': args.epochs,                                      # Maxm. no. of epochs to run   
        'gradient_clip_val': args.grad_clip,                           # Gradient clipping value
        'precision': args.precision,                                    # Precision to use for training
        'check_val_every_n_epoch': 1,                                   # No. of epochs to run validation
        'callbacks': [LearningRateMonitor(logging_interval='epoch'),    # Callbacks to use for training
                      EarlyStopping(monitor="val_loss", patience=10),
                      checkpoint_callback],
        'logger': comet_logger,                                         # Logger to use for training
    }

    if args.gpu_nodes > 1:
        trainer_args['strategy'] = args.dist_backend

    trainer = pl.Trainer(**trainer_args)

    # Fit the model to the training data using the Trainer's fit method.
    trainer.fit(spoof_trainer, dataloader)
    trainer.validate(spoof_trainer, dataloader)
    trainer.test(spoof_trainer, dataloader)


if __name__  == "__main__":
    parser = argparse.ArgumentParser(description="Train")

    # Train Device Hyperparameters
    parser.add_argument('-d', '--device', default='cuda', type=str, choices=['cpu', 'cuda', 'mps'],
                        help='device to use for training, default cuda')
    parser.add_argument('-g', '--gpu_nodes', default=1, type=int, help='number of gpus per node')
    parser.add_argument('-w', '--data_workers', default=8, type=int,
                        help='n data loading workers, default 8')
    parser.add_argument('-db', '--dist_backend', default='ddp', type=str, help='which distributed backend to use for aggregating multi-gpu train')

    # Dataset Base Directory
    parser.add_argument('--lcc_dir', 
                        required=True, type=str, 
                        help='Folder path to load data')
    
    # General Train Hyperparameters
    parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=32, type=int, help='size of batch')
    parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--grad_clip', default=0.6, type=float, help='gradient clipping value')
    parser.add_argument('--focal_gamma', default=2.0, type=float, help='gamma value for focal loss')
    parser.add_argument('--precision', default='16-mixed', type=str, help='precision')
    
    args = parser.parse_args()
    main(args)