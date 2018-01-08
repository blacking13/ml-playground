# AWS Sagemaker Docker Container

AWS Sagemaker allows train and execute multiple ML models including custom ones, but with some complications:
* Training jobs based on known engines(TensorFlow/MXNET) requires implementation of specific interfaces.
* Using custom Docker image for training assumes independent docker image for different training algorithms.

This set of utils allows execute custom ML training job via AWS Sagemaker having only single Docker image. Which allows use of GPU instance EC2 via for training ML.
Basically, Docker image will be independent of the training script and can be reused.
Sagemaker will execute training script which is hosted in S3. **Different training scripts controlled by different location of input data in S3 during training job scheduling.**

## How to use

There are two docker images available: Dockerfile.cpu, Dockerfile.gpu.
Docker files have pre-installed Python3, Keras with TensorFlow backend.

### Build docker file and upload to AWS ECR:

Save credentials in environment variables via:
```
aws configure
```

Build docker image and upload to AWS ECR:

CPU:
```
./build_and_upload.sh sagemaker-cpu Dockerfile.cpu
```

GPU:
```
./build_and_upload.sh sagemaker-gpu Dockerfile.gpu
```

build_and_upload.sh script will output ARN of Docker image in AWS ECR.

### Prepare input data and training script

Create and copy desired python script which contains ML model into S3 location. Script has to have name "train.py":
```
s3://[Bucket]/ml/input/script
```

Copy training data to S3:
```
s3://[Bucket]/ml/input/
```

Sagemaker will copy entire content of input data to local disk, including "scripts" folder.
Docker image includes "train" script, which will be executed by Sagemaker Training job.
"train" script will look into input folder and execute "input/data/scripts/train.py" downloaded from S3.

### Schedule training job

Training job can be scheduled from Sagemaker notebook instance:

Prepare imports and data location:
```python
import boto3
import re

import os
import numpy as np
import pandas as pd
from sagemaker import get_execution_role
import sagemaker as sage
from time import gmtime, strftime

role = get_execution_role()
sess = sage.Session()

bucket = [BUCKET]
folder = 'ml'

data_location = "s3://{}/{}/input".format(bucket, folder)
output_path = "s3://{}/{}/output".format(bucket, folder)

print ('Data Location: ', data_location)
print ('Output Location: ', output_path)
```

Execute training on CPU by specifying CPU image and EC2 instance type.
```python
account = sess.boto_session.client('sts').get_caller_identity()['Account']
region = sess.boto_session.region_name
image = '{}.dkr.ecr.{}.amazonaws.com/sagemaker-cpu'.format(account, region)

tree = sage.estimator.Estimator(image,
                       role, 1, 'ml.c4.2xlarge',
                       output_path=output_path,
                       sagemaker_session=sess)

tree.fit(data_location)
```

Training on GPU can be specified by appropriate Docker image and EC2 instance type of GPU family:
```python
account = sess.boto_session.client('sts').get_caller_identity()['Account']
region = sess.boto_session.region_name
image = '{}.dkr.ecr.{}.amazonaws.com/sagemaker-gpu'.format(account, region)

tree = sage.estimator.Estimator(image,
                       role, 1, 'ml.p2.xlarge',
                       output_path=output_path,
                       sagemaker_session=sess)

tree.fit(data_location)
```

In theory, Sagemaker notebook is not required and training job can be scheduled from command line. But following script didn't worked last time:
```bash
aws sagemaker create-training-job \
   --training-job-name "sagemaker-cpu" \
   --algorithm-specification TrainingImage=[ACCOUNT].dkr.ecr.us-west-2.amazonaws.com/sagemaker-cpu,TrainingInputMode=File \
   --role-arn arn:aws:iam::[ACCOUNT]:role/service-role/AmazonSageMaker-ExecutionRole-[ROLE ID] \
   --input-data-config ChannelName=training,DataSource={S3DataSource={S3DataType="S3Prefix",S3Uri="s3://[BUCKET]/ml/input"}} \
   --output-data-config S3OutputPath="s3://[BUCKET]/ml/output" \
   --resource-config InstanceType="ml.p3.2xlarge",InstanceCount=1,VolumeSizeInGB=1 \
   --stopping-condition MaxRuntimeInSeconds=3000
```

All output of the model will be stored in:
```
s3://[BUCKET]/ml/output
```

### Local testing

local_test includes sample train script for classification of cifar dataset.

Note: Train/test data is omitted. It can be downloaded and extracted to local_test/test_dir/input/data folder from https://www.cs.toronto.edu/~kriz/cifar.html

train.sh will start local docker instance and execute training from local_test/test_dir/input/data/script/train.py.

Sagemaker expects following folder layout:
* Input files: /opt/ml/input/data/[channel]
* Output files: /opt/ml/output.

Channel can be configured during job execution. By default: *training*

## Acknowledgments and references
* AWS Sagemaker tutorial: https://github.com/awslabs/amazon-sagemaker-examples/blob/master/advanced_functionality/scikit_bring_your_own/scikit_bring_your_own.ipynb
* Keras docker image: https://github.com/gw0/docker-keras
