import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as T

from dataloaders.GSVCitiesDataset import GSVCitiesDataset, GSVCitiesValDataset
# from . import PittsburgDataset
# from . import MapillaryDataset

from prettytable import PrettyTable

TYPE2LAMBDA = {
    'resize': lambda **kw: T.Resize(size=kw['size'], interpolation=T.InterpolationMode.BILINEAR),
    'randaug': lambda **kw: T.RandAugment(num_ops=kw['num_ops'], interpolation=T.InterpolationMode.BILINEAR),
    'totensor': lambda **kw: T.ToTensor(),
    'normalize': lambda **kw: T.Normalize(mean=kw['mean'], std=kw['std']),
    'centercrop': lambda **kw: T.CenterCrop(size=kw['size']),
    'pad': lambda **kw: T.Pad(size=kw['size'], fill=kw.get('fill', 0))
}

def build_transform_compose(transforms):
    to_compose = []
    for t in transforms:
        t = TYPE2LAMBDA[t['type']](**t.get('kwargs', {}))
        to_compose.append(t)
    return T.Compose(to_compose)
    
        

class GSVCitiesDataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size=32,
                 img_per_place=4,
                 min_img_per_place=4,
                 shuffle_all=False,
                 image_size=(480, 640),
                 num_workers=4,
                 show_data_stats=True,
                 batch_sampler=None,
                 random_sample_from_each_place=True,
                 train_anno=None,
                 query_anno=[],
                 ref_anno=[],
                 base_path=None,
                 train_transform=[],
                 vals_transforms=[],
                 train_transform_e=[]
                 ):
        super().__init__()
        self.batch_size = batch_size
        self.img_per_place = img_per_place
        self.min_img_per_place = min_img_per_place
        self.shuffle_all = shuffle_all
        self.image_size = image_size
        self.num_workers = num_workers
        self.batch_sampler = batch_sampler
        self.show_data_stats = show_data_stats
        self.random_sample_from_each_place = random_sample_from_each_place
        self.save_hyperparameters() # save hyperparameter with Pytorch Lightening

        self.train_anno = train_anno
        self.query_anno = query_anno
        self.ref_anno = ref_anno
        self.base_path = base_path
        # import pdb; pdb.set_trace()
        assert len(query_anno) == len(vals_transforms) == len(ref_anno), 'data not match with transform'
        self.train_transform = build_transform_compose(train_transform)
        self.train_transform_e = build_transform_compose(train_transform_e)
        self.valid_transforms = [build_transform_compose(vals_transform) for vals_transform in vals_transforms]

        self.train_loader_config = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'drop_last': False,
            'pin_memory': True,
            'shuffle': self.shuffle_all}
            

        self.valid_loader_config = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers//2,
            'drop_last': False,
            'pin_memory': True,
            'shuffle': False}

    def setup(self, stage):
        if stage == 'fit':
            self.reload()
            # load validation sets (pitts_val, msls_val, ...etc)
            self.val_set_names = self.query_anno
            self.val_datasets = [GSVCitiesValDataset(
            img_per_place=self.img_per_place,
            min_img_per_place=self.min_img_per_place,
            random_sample_from_each_place=self.random_sample_from_each_place,
            transform=self.train_transform,
            base_path=self.base_path,
            query_anno=q,
            ref_anno=r) for q, r in zip(self.query_anno, self.ref_anno)]
            if self.show_data_stats:
                self.print_stats()

    def reload(self):
        self.train_dataset = GSVCitiesDataset(
            img_per_place=self.img_per_place,
            min_img_per_place=self.min_img_per_place,
            random_sample_from_each_place=self.random_sample_from_each_place,
            transform=self.train_transform,
            transform_e=self.train_transform_e,
            base_path=self.base_path,
            train_anno=self.train_anno)

    def train_dataloader(self):
        self.reload()
        return DataLoader(dataset=self.train_dataset, **self.train_loader_config)

    def val_dataloader(self):
        val_dataloaders = []
        for val_dataset in self.val_datasets:
            val_dataloaders.append(DataLoader(
                dataset=val_dataset, **self.valid_loader_config))
        return val_dataloaders

    def print_stats(self):
        print()  # print a new line
        table = PrettyTable()
        table.field_names = ['Data', 'Value']
        table.align['Data'] = "l"
        table.align['Value'] = "l"
        table.header = False
        table.add_row(["# of places", f'{self.train_dataset.__len__()}'])
        table.add_row(["# of images", f'{self.train_dataset.total_nb_images}'])
        print(table.get_string(title="Training Dataset"))
        print()

        table = PrettyTable()
        table.field_names = ['Data', 'Value']
        table.align['Data'] = "l"
        table.align['Value'] = "l"
        table.header = False
        # table.add_row(["# of places", f'{self.train_dataset.__len__()}'])
        print(table.get_string(title="Validation Datasets"))
        print()

        table = PrettyTable()
        table.field_names = ['Data', 'Value']
        table.align['Data'] = "l"
        table.align['Value'] = "l"
        table.header = False
        table.add_row(
            ["Batch size (PxK)", f"{self.batch_size}x{self.img_per_place}"])
        table.add_row(
            ["# of iterations", f"{self.train_dataset.__len__()//self.batch_size}"])
        table.add_row(["Image size", f"{self.image_size}"])
        print(table.get_string(title="Training config"))
