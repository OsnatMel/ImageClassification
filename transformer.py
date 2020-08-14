from config import *
from imports import *

class topCrop(object):
    """Crop upper rectengle of the image to a given size.
    """
    def __init__(self, size=224):
        self.size = size
    def __call__(self, image):
        w, h = image.size
        off = int(224-w)/2
        out_image = image.crop((-off, 0, self.size-off, self.size))
        return out_image

class transformer():
    def __init__(self,resize=224,keep_AR=False,crop=False,topCrop=False,flip=False,normalize=False):
        assert isinstance(resize, (int,bool)), "resize should be bool or int"
        assert isinstance(keep_AR, bool), "keep_AR should be bool"
        assert isinstance(crop, (int,bool)), "crop should be bool or int"
        assert isinstance(topCrop, bool), "topCrop should be bool"
        assert isinstance(flip, bool), "horizontal_flip should be bool"
        assert isinstance(normalize, bool), "normalize should be bool"
        self.resize = resize
        self.keep_AR = keep_AR
        self.crop = crop
        self.topCrop = topCrop
        self.flip = flip
        self.normalize = normalize
        self.steps = [transforms.ToTensor()]
        self.is_train = None
        self.train_transformer = None
        self.test_transformer = None
        self.append_steps()
        self.create_transformer()

    def append_steps(self):
        steps = []
        if self.resize and self.keep_AR:
            steps.append(transforms.Resize(self.resize))
        if self.resize and not self.keep_AR:
            steps.append(transforms.Resize((self.resize,self.resize)))
        if self.crop:
            steps.append(transforms.CenterCrop(size=224))
        if self.topCrop:
            steps.append(topCrop())
        if self.flip and self.is_train:
            steps.append(transforms.RandomHorizontalFlip(p=0.5))
        steps.append(transforms.ToTensor())
        if self.normalize:
            steps.append(transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]))

        self.steps = steps

    def compose_transformer(self,is_train=False):
        self.is_train = is_train
        self.append_steps()
        transform = transforms.Compose(self.steps)
        return transform

    def create_train_transformer(self):
        self.train_transformer = self.compose_transformer(is_train=True)
        return

    def create_test_transformer(self):
        self.test_transformer = self.compose_transformer(is_train=False)
        return

    def create_transformer(self):
        self.create_test_transformer()
        self.create_train_transformer()
        self.document_transformer()
        return

    def plot_transformed_img(self,img_path,aws=True):
        if aws:
            s3 = boto3.resource('s3')
            obj = s3.Object(BUCKETNAME, img_path)
            image = Image.open(obj.get()['Body'])
        else:
            image = Image.open(img_path)
        plt.subplot(1,2,1)
        plt.title('Before')
        plt.imshow(image)
        plt.axis('off')
        t_image = self.test_transformer(image)
        np_img = np.transpose(t_image, (1, 2, 0))
        plt.subplot(1,2,2)
        plt.imshow(np_img);
        plt.title('After')
        plt.axis('off')

    def document_transformer(self):
#         f = open("runs/"+TB_DIR+"/model_description.txt","a")
        f = open(TB_DIR+"/model_description.txt","a")
        f.write("-- transformer steps \r\n")
        for s in self.steps:
            f.write('   '+str(s)+"\r\n")
        f.close()
