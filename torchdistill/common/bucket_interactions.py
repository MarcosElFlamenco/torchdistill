import boto3
import torch
from tqdm import tqdm


def empty_s3_bucket(bucket_name):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    bucket.objects.all().delete()


def checkpoint_exists(bucket_name, checkpoint_key):
    s3 = boto3.client('s3')
    try:
        s3.head_object(Bucket=bucket_name, Key=checkpoint_key)
        return True
    except s3.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        else:
            raise

def download_checkpoint(bucket_name, checkpoint_key, local_file_path):
    s3 = boto3.client('s3')
    try:
        s3.download_file(bucket_name, checkpoint_key, local_file_path)
        return 0
    except e:
        return 1

def upload_checkpoint(local_file_path, bucket_name, checkpoint_key):
    s3 = boto3.client('s3')
    s3.upload_file(local_file_path, bucket_name, checkpoint_key)

def save_checkpoint(epoch, model, optimizer, bucket_name, checkpoint_key):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, 'checkpoint.pth')
    upload_checkpoint('checkpoint.pth', bucket_name, checkpoint_key)

def List_s3_objects(bucket_name, prefix=None):

    s3_client = boto3.client('s3')
    objects = []
    paginator = s3_client.get_paginator('list_objects_v2')
    operation_parameters = {'Bucket': bucket_name}
    if prefix:
        operation_parameters['Prefix'] = prefix

    for page in paginator.paginate(**operation_parameters):
        if 'Contents' in page:
            for obj in page['Contents']:
                objects.append(obj['Key'])
    
    return objects

def load_checkpoint(bucket_name, checkpoint_key_prefix, device):
    local_file_path = 'checkpoint.pth'
    contents = List_s3_objects(bucket_name, prefix=checkpoint_key_prefix)

    if contents:
        full_prefix = checkpoint_key_prefix + "_"
        versions = [int(s.removeprefix(full_prefix).removesuffix("K.pth")) for s in contents] 
        most_recent_version = max(versions)
        checkpoint_key = checkpoint_key_prefix + "_" + str(most_recent_version) + "K.pth"
        print('A checkpoint does in fact exist, now loading...')
        print(f"Resuming training from checkpoint {checkpoint_key} in bucket {bucket_name}")
        try:
            download_checkpoint(bucket_name, checkpoint_key, local_file_path)
            checkpoint = torch.load(local_file_path, map_location=device)
            return checkpoint
        except Exception as e:
            print(f"Got the following error {e} so we're gonna start from scratch")
            return None
    else:
        return None


def download_bins_from_s3_with_progress(bucket_name, object_name, file_name):
    s3 = boto3.client('s3')

    # Get the size of the file from S3 to set up the progress bar
    response = s3.head_object(Bucket=bucket_name, Key=object_name)
    total_size_in_bytes = response['ContentLength']

    # Open the file locally in binary write mode
    with open(file_name, 'wb') as f:
        # Use tqdm to create the progress bar
        with tqdm(total=total_size_in_bytes, unit='B', unit_scale=True, desc=f"Downloading {object_name}") as bar:
            # Download the file in chunks and update the progress bar
            s3.download_fileobj(Bucket=bucket_name, Key=object_name, Fileobj=f, Callback=lambda bytes_transferred: bar.update(bytes_transferred))