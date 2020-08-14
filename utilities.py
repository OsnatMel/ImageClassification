
from imports import *

def create_data_csv(aws=False):
    '''
    The function create and saves a csv containing image path and labels.
    Input path parameters are taken from config file
    '''
    if aws:
        s3 = boto3.client("s3")

        paths_valid = []
        paths_invalid = []

        resp = s3.list_objects_v2(Bucket=BUCKETNAME,Prefix ='Training_Data/Valid/')
        for cont in resp['Contents']:
            fname = cont['Key']
            if fname[-3:]=='png' or fname[-3:]=='jpg':
                paths_valid.append(fname)

        resp = s3.list_objects_v2(Bucket=BUCKETNAME,Prefix ='Training_Data/Invalid/')
        for cont in resp['Contents']:
            fname = cont['Key']
            if fname[-3:]=='png' or fname[-3:]=='jpg':
                paths_invalid.append(fname)

    else:
        paths_valid = [f'{ROOT_DIR}Valid/{fname}' for fname in os.listdir(ROOT_DIR+'Valid/') if fname not in ['.DS_Store']]
        paths_invalid = [f'{ROOT_DIR}Invalid/{fname}' for fname in os.listdir(ROOT_DIR+'Invalid/') if fname not in ['.DS_Store']]

    data = pd.DataFrame()
    data['path'] = paths_valid+paths_invalid
    data['target'] = np.concatenate((np.ones(len(paths_valid)),np.zeros(len(paths_invalid))),axis=0) 
    data.to_csv(FNAME,index=False)
    return