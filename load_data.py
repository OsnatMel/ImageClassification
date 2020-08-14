from imports import *

torch.manual_seed(SEED)
np.random.seed(SEED)

class MyDataset(Dataset):
    '''
    Custom data set to be used for pytorch DataLoader.
    Init inputs:
    ~~~~~~~~~~~
    - pd.df with <path_col> for image path column name and <target_col> for label column name
    - root_dir to read images from using <path_col>
    - transform
    - path_col name
    - label_col name
    '''
    def __init__(self,df,root_dir,transform=None,path_col = 'path',target_col='target',aws=True):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.count = len(df)
        self.path_col = path_col
        self.target_col = target_col
        self.aws=aws
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.df.loc[idx, self.path_col]
        img_path = img_name#os.path.join(self.root_dir,img_name)
        if self.aws:
            s3 = boto3.resource('s3')
            image = Image.open(s3.Object(BUCKETNAME, img_path).get()['Body'])
        else:
            image = Image.open(img_path)
        label = int(self.df.loc[idx, self.target_col])
        sample = {
            'image' : image,
            'label' :label,
            'path': img_name,
        }
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        return sample

    def __len__(self):
        return self.count

############################################

def create_loaders(transform,aws=True):
    '''
    This create_loaders function create and return a MyDataset object built of data existing in ROOT_DIT
    Config parameters:
    ~~~~~~~~~~~~~~~~~~
    -- ROOT_DIR+FNAME for data descriptor csv
    -- VAL_SIZE : validation set ratio
    -- TEST_SIZE : validation set ratio
    -- BATCH_SIZE
    Input params:
    ~~~~~~~~~~~~~
    -- transform : a transformer with train_transformer and test_transformer attributes
    Output:
    ~~~~~~
    -- loaders : a dict with {'train' : train_loader, 'val' : val_loader, 'test' : test_loader}
    '''
    #read images into pd df and split to train-val-test
    df = pd.read_csv(FNAME)
    num_images = len(df)
    indices = np.arange(num_images)
    np.random.shuffle(indices)

    val_split = int(VAL_SIZE * num_images)
    test_split = int((VAL_SIZE+TEST_SIZE) * num_images)

    val_idx, test_idx, train_idx = indices[:val_split], indices[val_split:test_split], indices[test_split:]

    #create train-val-test df for loaders
    train_df = df.loc[train_idx].reset_index(drop=True)
    val_df = df.loc[val_idx].reset_index(drop=True)
    test_df = df.loc[test_idx].reset_index(drop=True)

    #create and print data statistics
    train_stat = train_df.groupby('target').count()
    val_stat = val_df.groupby('target').count()
    test_stat = test_df.groupby('target').count()

    print(f'Train: total {len(train_idx)}, valid: {train_stat.loc[1].values.item()}, invalid: {train_stat.loc[0].values.item()}')
    print(f'Val:   total {len(val_df)},  valid: {val_stat.loc[1].values.item()},  invalid: {val_stat.loc[0].values.item()}')
    print(f'Test:  total {len(test_df)},  valid: {test_stat.loc[1].values.item()},  invalid: {test_stat.loc[0].values.item()}')

    #creating the dataset object
    train_set = MyDataset(train_df,root_dir=ROOT_DIR,transform=transform.train_transformer,path_col='path',target_col='target',aws=aws)
    val_set = MyDataset(val_df,root_dir=ROOT_DIR,transform=transform.test_transformer,path_col='path',target_col='target',aws=aws)
    test_set = MyDataset(test_df,root_dir=ROOT_DIR,transform=transform.test_transformer,path_col='path',target_col='target',aws=aws)

    # #Create DataLoader iterators
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    loaders = {'train' : train_loader,
               'val' : val_loader,
               'test' : test_loader}
    return loaders
