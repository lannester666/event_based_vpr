import argparse
import yaml

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from torch.optim import lr_scheduler, optimizer
import utils

from dataloaders.GSVCitiesDataloader import GSVCitiesDataModule
from models import helper

import mlflow
from pytorch_lightning.loggers import MLFlowLogger


class VPRModel(pl.LightningModule):
    """This is the main model for Visual Place Recognition
    we use Pytorch Lightning for modularity purposes.

    Args:
        pl (_type_): _description_
    """

    def __init__(self,
                 # ---- Backbone
                 backbone_arch='resnet50',
                 pretrained=True,
                 layers_to_freeze=1,
                 layers_to_crop=[],

                 # ---- Aggregator
                 agg_arch='ConvAP',  # CosPlace, NetVLAD, GeM
                 agg_config={},

                 # ---- Train hyperparameters
                 lr=0.03,
                 optimizer='sgd',
                 weight_decay=1e-3,
                 momentum=0.9,
                 warmpup_steps=500,
                 milestones=[5, 10, 15],
                 lr_mult=0.3,

                 # ----- Loss
                 loss_name='MultiSimilarityLoss',
                 miner_name='MultiSimilarityMiner',
                 miner_margin=0.1,
                 faiss_gpu=False,
                 one4two=False,
                 bimod=False
                 ):
        super().__init__()
        self.encoder_arch = backbone_arch
        self.pretrained = pretrained
        self.layers_to_freeze = layers_to_freeze
        self.layers_to_crop = layers_to_crop

        self.agg_arch = agg_arch
        self.agg_config = agg_config

        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.warmpup_steps = warmpup_steps
        self.milestones = milestones
        self.lr_mult = lr_mult

        self.loss_name = loss_name
        self.miner_name = miner_name
        self.miner_margin = miner_margin

        self.save_hyperparameters()  # write hyperparams into a file

        self.loss_fn = utils.get_loss(loss_name)
        self.miner = utils.get_miner(miner_name, miner_margin)
        # we will keep track of the % of trivial pairs/triplets at the loss level
        self.batch_acc = []

        self.faiss_gpu = faiss_gpu
        if bimod and not one4two:
            self.model_i = self.build_model(backbone_arch, pretrained, layers_to_freeze, layers_to_crop, agg_arch, agg_config)
            self.model_e = self.build_model(backbone_arch, pretrained, layers_to_freeze, layers_to_crop, agg_arch, agg_config)
        else:
            self.model_i = self.build_model(backbone_arch, pretrained, layers_to_freeze, layers_to_crop, agg_arch, agg_config)
            self.model_e = self.model_i
        # ----------------------------------
        # get the backbone and the aggregator
       

    def build_model(self, backbone_arch, pretrained, layers_to_freeze, layers_to_crop, agg_arch, agg_config):
        ex = torch.nn.Conv2d(1, 3, 1, bias=False)
        ex.weight = torch.nn.Parameter(torch.ones(3, 1, 1, 1))
        backbone = helper.get_backbone(
            backbone_arch, pretrained, layers_to_freeze, layers_to_crop)
        aggregator = helper.get_aggregator(agg_arch, agg_config)
        return torch.nn.Sequential(ex, backbone, aggregator)
    
    # the forward pass of the lightning model
    def forward(self, *x):
        if len(x) == 1:
            x = x[0]
            BS, N, ch, h, w = x.shape
            x = x.view(BS*N, ch, h, w)
            return self.model_i(x)
        else:
            BS, N, ch, h, w = x[0].shape
            x0 = x[0].view(BS*N, ch, h, w)
            x1 = x[1].view(BS*N, ch, h, w)
            return self.model_i(x0), self.model_e(x1)

    # configure the optimizer
    def configure_optimizers(self):
        if self.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(),
                                        lr=self.lr,
                                        weight_decay=self.weight_decay,
                                        momentum=self.momentum)
        elif self.optimizer.lower() == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(),
                                          lr=self.lr,
                                          weight_decay=self.weight_decay)
        elif self.optimizer.lower() == 'adam':
            optimizer = torch.optim.AdamW(self.parameters(),
                                          lr=self.lr,
                                          weight_decay=self.weight_decay)
        else:
            raise ValueError(
                f'Optimizer {self.optimizer} has not been added to "configure_optimizers()"')
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, milestones=self.milestones, gamma=self.lr_mult)
        return [optimizer], [scheduler]

    # configure the optizer step, takes into account the warmup stage
    def optimizer_step(self,  epoch, batch_idx,
                       optimizer, optimizer_idx, optimizer_closure,
                       on_tpu, using_native_amp, using_lbfgs):
        # warm up lr
        if self.trainer.global_step < self.warmpup_steps:
            lr_scale = min(
                1., float(self.trainer.global_step + 1) / self.warmpup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.lr
        optimizer.step(closure=optimizer_closure)

    #  The loss function call (this method will be called at each training iteration)
    def loss_function(self, descriptors, labels):
        # we mine the pairs/triplets if there is an online mining strategy
        if isinstance(descriptors, torch.Tensor):
            if self.miner is not None:
                miner_outputs = self.miner(descriptors, labels)
                loss = self.loss_fn(descriptors, labels, miner_outputs)

                nb_samples = descriptors.shape[0]
                nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
                batch_acc = 1.0 - (nb_mined/nb_samples)

            else:  # no online mining
                loss = self.loss_fn(descriptors, labels)
                batch_acc = 0.0
                if type(loss) == tuple:
                    loss, batch_acc = loss
        else:
            assert len(descriptors) == 2
            if self.miner is not None:
                miner_outputs = self.miner(descriptors[0], labels)
                loss = self.loss_fn(descriptors[0], labels, miner_outputs)
                nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
                miner_outputs = self.miner(descriptors[1], labels)
                loss += self.loss_fn(descriptors[1], labels, miner_outputs)

                nb_samples = descriptors[0].shape[0] * 2
                nb_mined += len(set(miner_outputs[0].detach().cpu().numpy()))
                batch_acc = 1.0 - (nb_mined/nb_samples)

            else:  # no online mining
                loss = self.loss_fn(descriptors[0], labels)
                loss += self.loss_fn(descriptors[1], labels)
                batch_acc = 0.0
                if type(loss) == tuple:
                    loss, batch_acc = loss

        # keep accuracy of every batch and later reset it at epoch start
        self.batch_acc.append(batch_acc)
        # log it
        self.log('b_acc', sum(self.batch_acc) /
                 len(self.batch_acc), prog_bar=True, logger=True)
        return loss

    # This is the training step that's executed at each iteration
    def training_step(self, batch, batch_idx):
        # places, labels = batch
        labels = batch[-1]
        places = batch[:-1]
        places = (places[0], ) if len(places) == 1 else tuple(places)
        labels = labels.view(-1)

        # Feed forward the batch to the model
        # Here we are calling the method forward that we defined above
        descriptors = self(*places)
        # Call the loss_function we defined above
        loss = self.loss_function(descriptors, labels)

        self.log('loss', loss.item(), logger=True)
        return {'loss': loss}

    # This is called at the end of eatch training epoch
    def training_epoch_end(self, training_step_outputs):
        # we empty the batch_acc list for next epoch
        self.batch_acc = []

    # For validation, we will also iterate step by step over the validation set
    # this is the way Pytorch Lghtning is made. All about modularity, folks.
    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        places, _ = batch
        places = (places.reshape(1, *places.shape), )
        # calculate descriptors
        descriptors = self(*places)
        return descriptors.detach().cpu()

    def validation_epoch_end(self, val_step_outputs):
        """this return descriptors in their order
        depending on how the validation dataset is implemented 
        for this project (MSLS val, Pittburg val), it is always references then queries
        [R1, R2, ..., Rn, Q1, Q2, ...]
        """
        dm = self.trainer.datamodule
        # The following line is a hack: if we have only one validation set, then
        # we need to put the outputs in a list (Pytorch Lightning does not do it presently)
        if len(dm.val_datasets) == 1:  # we need to put the outputs in a list
            val_step_outputs = [val_step_outputs]

        for i, (val_set_name, val_dataset) in enumerate(zip(dm.val_set_names, dm.val_datasets)):
            feats = torch.concat(val_step_outputs[i], dim=0)
            num_references = val_dataset.num_references
            positives = val_dataset.label
            r_list = feats[: num_references]
            q_list = feats[num_references:]
            pitts_dict = utils.get_validation_recalls(r_list=r_list,
                                                      q_list=q_list,
                                                      k_values=[
                                                          1, 5, 10, 15, 20, 50, 100],
                                                      gt=positives,
                                                      print_results=True,
                                                      dataset_name=val_set_name,
                                                      faiss_gpu=self.faiss_gpu
                                                      )
            del r_list, q_list, feats, num_references, positives

            val_set_name = val_set_name.replace('/', '.')
            self.log(f'{val_set_name}/R1',
                     pitts_dict[1], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R5',
                     pitts_dict[5], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R10',
                     pitts_dict[10], prog_bar=False, logger=True)
        print('\n\n')


def yaml2config(fn):
    with open(fn, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
        f.close()
    seed = config.get('seed', 202305)
    assert 'data' in config, 'you have to specify the data config'
    assert 'model' in config, 'you have to specify the model config'
    assert 'trainer' in config, 'you have to specify the trainer config'
    datamodule_config = config['data']
    model_config = config['model']
    trainer_config = config['trainer']
    return datamodule_config, model_config, trainer_config, seed
if __name__ == '__main__':
    parser = argparse.ArgumentParser('parser for vpr of CS284')
    parser.add_argument('--config', required=True, type=str)
    args = parser.parse_args()
    datamodule_config, model_config, trainer_config, seed = yaml2config(args.config)
    pl.utilities.seed.seed_everything(seed=seed, workers=True)

    datamodule = GSVCitiesDataModule(**datamodule_config)
    model = VPRModel(**model_config)
    mlf_logger = MLFlowLogger(experiment_name="lightning_logs", tracking_uri="file:./mlruns")
    trainer_config.update({'logger': mlf_logger})
    trainer = pl.Trainer(**trainer_config)
    trainer.fit(model=model, datamodule=datamodule)
