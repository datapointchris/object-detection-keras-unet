import boto3
import config
from pathlib import Path
from tqdm import tqdm


if config.main.ENVIRONMENT == 'production':
    s3 = boto3.resource('s3')
else:
    session = boto3.Session(profile_name='chrisbirchlive_chris-admin')
    s3 = session.resource('s3')

bucket = s3.Bucket(config.main.s3_bucket)


def download_s3_objects(bucket: s3.Bucket, object_path: Path):
    for object in tqdm(bucket.objects.filter(Prefix=str(object_path)).all()):
        bucket.download_file(object.key, object.key)


Path.mkdir(config.main.train_image_path, parents=True, exist_ok=True)
Path.mkdir(config.main.train_mask_path, parents=True, exist_ok=True)

download_s3_objects(bucket, config.main.train_image_path)
download_s3_objects(bucket, config.main.train_mask_path)
