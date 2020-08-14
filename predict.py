from imports import *

def predict(image_path, model, transform,aws=False):
    model.eval()
    if aws:
        s3 = boto3.resource('s3')
        image = Image.open(s3.Object(BUCKETNAME, image_path).get()['Body'])
    else:
        image = Image.open(image_path)
    image_tensor = transform.test_transformer(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(DEVICE)
    output = model(image_tensor)
    index = output.argmax().item()
    return CLASSES[index]
