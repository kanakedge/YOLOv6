import boto3
from botocore.exceptions import ClientError
import os

extensions = ["jpg","jpeg","png","mp4","MOV","avi"]

class aws_util:
    def __init__(self):
        self.s3_client = boto3.client('s3', region_name="ap-south-1")
        self.location = {'LocationConstraint': "ap-south-1"}
        self.s3_resource = boto3.resource('s3')
    def create_bucket(self, bucket_name, region=None):
        response = self.s3_client.create_bucket(Bucket=bucket_name,CreateBucketConfiguration=self.location)
        if response is None:
            return True
    def check_bucket(self, bucket_name):
        if self.s3_resource.Bucket(bucket_name).creation_date is None:
            return False
        return True
    def upload_file(self, file_name, bucket_name, object_name=None):
        if object_name is None:
            object_name = file_name
        response = self.s3_client.upload_file(file_name, bucket_name, object_name)
        if response is None:
            return True
    def download_Directory(self, bucket_name,remoteDirectoryName):
        print(remoteDirectoryName)
        self.bucket = self.s3_resource.Bucket(bucket_name) 
        for object in self.bucket.objects.filter(Prefix = remoteDirectoryName):
            print(object.key)
            extension = object.key.split(".")[-1]
            print(extension)
            if extension in extensions:
                save_path = remoteDirectoryName + "/" + "Raw_Data" + object.key.split(remoteDirectoryName)[1]
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                response = self.bucket.download_file(object.key,save_path)
                response=None
                if response is None:
                    continue
            else:
                print("Skipping: ",object.key)
    def download_file(self, bucket_name, filename, download_path):
        response = self.s3_client.download_file(bucket_name, filename, download_path)
        if response is None:
            return True
def main():
    aws_U = aws_util()
    #response = aws_U.download_Directory("edgeneural-enrollment-demo","DemoCompany-100")
    response = aws_U.download_file("enap-train-data","classification_data.zip","data_cfg/data_url.zip")
    print(response)
if __name__=="__main__":
    main()
