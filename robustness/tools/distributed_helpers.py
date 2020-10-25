class DistributedLoader():
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.args = args
    
    def loaders(self, world_size, rank):
        args = {**self.args, 'distributed_args': (world_size, rank)}
        return self.dataset.make_loaders(**args)

class DistributedStore():
    def __init__(self, setup_fn, args):
        self.setup_fn = setup_fn 
        self.args = args
    
    def setup(self):
        return self.setup_fn(self.args)