pip install pandas==1.4.2
pip install sentence-transformers==2.2.2
pip install rich==12.2.0
pip install pyarrow==7.0.0

mkdir data
wget https://fm-data-tasks.s3.us-west-1.amazonaws.com/datasets.tar.gz -P data
tar xvf data/datasets.tar.gz -C data/