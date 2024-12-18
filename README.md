
# Data Driven Approach to Payment Fraud Detection

Online payment fraud is a growing issue affecting businesses and consumers. Detecting fraudulent transactions is complex due to the large volume and velocity of transaction data. This project proposes a data-driven approach using machine learning and big data technologies. The framework uses Hadoop for storage and processing, HBase for data management, and machine learning models for fraud detection. Tested on a real-life dataset, the system efficiently flags suspicious transactions, offering a toolkit for financial losses reduction and transaction security enhancement.

## Higlights

🚀 Mapping Fraudulent Activity Across the U.S.
We visualized fraudulent transaction hotspots on an interactive U.S. map using precise latitude and longitude data. This map highlights concentrated fraudulent activities, especially in and around metro cities, providing valuable insights into geographic fraud trends. The visualization integrates seamlessly into our Streamlit app, offering an intuitive and aesthetically appealing user experience.

🚀 Dynamic Database-Driven Analysis
Our app dynamically integrates with two databases, allowing users to select their preferred data source. This flexibility enables analysis and visualization of fraud data based on the chosen database. Additionally, we successfully implemented CRUD operations, ensuring that any updates made are reflected in the HBase backend, enhancing the app’s functionality and data integrity.

## Questions (for each team member)


1. Aditya Thakare

## Question 1: "Is there a correlation between the customer age and the likelihood of fraud?" 

Analysis can be found in Dataset1_50608812_50604538_50606796_project.ipynb at line 57-58

## Question 2: Do certain product categories have a higher likelihood of fraudulent transactions?
Analysis can be found in Dataset2_50608812_50604538_50606796_project.ipynb at line 9-10

2. Onkar Ramade

## Question 3: What are demographic factors, including but not limited to account_age and location, that signal fraudulent e-commerce transactions?
Analysis can be found in Dataset1_50608812_50604538_50606796_project.ipynb at line 53-55

## Question 4: Can Transaction and Cardholder Geographic Location Predict Fraudulent Transactions?
Analysis can be found in Dataset2_50608812_50604538_50606796_project.ipynb at line 5-6


3. Sourabh Kodag

## Question 5 : "Do fraudulent transactions vary by hour, and can specific times of the day indicate a higher likelihood of fraudulent activities?"

Analysis can be found in Dataset1_50608812_50604538_50606796_project.ipynb at line 63-65

## Question 6: Is there a correlation between fraudulent transactions and locations near metro cities?
Analysis can be found in Dataset2_50608812_50604538_50606796_project.ipynb at line 7-8


## Tools used

Below are the list of tools used: 

- Python
- Hadoop
- HBase
- Numpy
- Pandas
- Scikit-Learn
- Matplotlib
- Streamlit

```bash
import zipfile
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import folium
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, classification_report, roc_auc_score,
                             confusion_matrix, roc_curve, precision_recall_curve,
                             auc, precision_score, recall_score, f1_score, log_loss)
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
```




## Installations

Follow below process for installing Hadoop on Linux

1. Hadoop

Prerequisite Test
=============================
sudo apt update
sudo apt install openjdk-8-jdk -y

java -version; javac -version
sudo apt install openssh-server openssh-client -y
sudo adduser hdoop
su - hdoop
ssh-keygen -t rsa -P '' -f ~/.ssh/id_rsa
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
chmod 0600 ~/.ssh/authorized_keys
ssh localhost

Downloading Hadoop 
===============================
wget https://dlcdn.apache.org/hadoop/common/hadoop-3.3.6/hadoop-3.3.6.tar.gz
tar xzf hadoop-3.3.6.tar.gz


Editng 6 important files
=================================
1st file
===========================
sudo nano .bashrc -  here you might face issue saying hdoop is not sudo user 
if this issue comes then
su - rootuser
sudo adduser hdoop sudo

sudo nano .bashrc
#Add below lines in this file

#Hadoop Related Options
export HADOOP_HOME=/home/hdoop/hadoop-3.3.6
export HADOOP_INSTALL=$HADOOP_HOME
export HADOOP_MAPRED_HOME=$HADOOP_HOME
export HADOOP_COMMON_HOME=$HADOOP_HOME
export HADOOP_HDFS_HOME=$HADOOP_HOME
export YARN_HOME=$HADOOP_HOME
export HADOOP_COMMON_LIB_NATIVE_DIR=$HADOOP_HOME/lib/native
export PATH=$PATH:$HADOOP_HOME/sbin:$HADOOP_HOME/bin
export HADOOP_OPTS"-Djava.library.path=$HADOOP_HOME/lib/nativ"


source ~/.bashrc

2nd File
============================
sudo nano $HADOOP_HOME/etc/hadoop/hadoop-env.sh

#Add below line in this file in the end

export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64

3rd File
===============================
sudo nano $HADOOP_HOME/etc/hadoop/core-site.xml

#Add below lines in this file(between "<configuration>" and "<"/configuration>")
   <property>
        <name>hadoop.tmp.dir</name>
        <value>/home/hdoop/tmpdata</value>
        <description>A base for other temporary directories.</description>
    </property>
    <property>
        <name>fs.default.name</name>
        <value>hdfs://localhost:9000</value>
        <description>The name of the default file system></description>
    </property>

4th File
====================================
sudo nano $HADOOP_HOME/etc/hadoop/hdfs-site.xml

#Add below lines in this file(between "<configuration>" and "<"/configuration>")


<property>
  <name>dfs.data.dir</name>
  <value>/home/hdoop/dfsdata/namenode</value>
</property>
<property>
  <name>dfs.data.dir</name>
  <value>/home/hdoop/dfsdata/datanode</value>
</property>
<property>
  <name>dfs.replication</name>
  <value>1</value>
</property>



5th File
================================================

sudo nano $HADOOP_HOME/etc/hadoop/mapred-site.xml

#Add below lines in this file(between "<configuration>" and "<"/configuration>")

<property>
  <name>mapreduce.framework.name</name>
  <value>yarn</value>
</property>

6th File
==================================================
sudo nano $HADOOP_HOME/etc/hadoop/yarn-site.xml

#Add below lines in this file(between "<configuration>" and "<"/configuration>")

<property>
  <name>yarn.nodemanager.aux-services</name>
  <value>mapreduce_shuffle</value>
</property>
<property>
  <name>yarn.nodemanager.aux-services.mapreduce.shuffle.class</name>
  <value>org.apache.hadoop.mapred.ShuffleHandler</value>
</property>
<property>
  <name>yarn.resourcemanager.hostname</name>
  <value>127.0.0.1</value>
</property>
<property>
  <name>yarn.acl.enable</name>
  <value>0</value>
</property>
<property>
  <name>yarn.nodemanager.env-whitelist</name>
  <value>JAVA_HOME,HADOOP_COMMON_HOME,HADOOP_HDFS_HOME,HADOOP_CONF_DIR,CLASSPATH_PERPEND_DISTCACHE,HADOOP_YARN_HOME,HADOOP_MAPRED_HOME</value>
</property>


Launching Hadoop
==================================
hdfs namenode -format


./start-dfs.sh






Below are the instruction for installing HBase

Install HBASE 

Download the 2.6.1/ tar file.

Go to the home directory and access the .bashrc file
sudo nano .bashrc

Write the following at the end of the bashrc file
#HBASE VARIABLES
export HBASE_HOME=/home/parallels/hbase-2.6.1
export PATH=$PATH:$HBASE_HOME/bin
#HBASE VARIABLES END

Then, implement the .bashrc file using the below command:
source ~/.bashrc

Now, inside the hbase-2.4.9, go to the conf folder, inside that go to
hbase-site.xml

<property>
    <name>hbase.cluster.distributed</name>
    <value>true</value>
  </property>
  <property>
    <name>hbase.tmp.dir</name>
    <value>./tmp</value>
  </property>
  <property>
    <name>hbase.unsafe.stream.capability.enforce</name>
    <value>false</value>
  </property>
  <property>
    <name>hbase:rootdir</name>
    <value>hdfs://localhost:9000</value>
  </property>
</configuration>


Now open hbase-env.sh in the same file and uncomment the JAVA_HOME line. and put your location.
# The java implementation to use.  Java 1.8+ required.
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-arm64


Make sure to do the following: First, hit 

start-all.sh

Then, if it runs successfully when you run jps, you should see the following output.
jps

Now, go to the conf directory in the base folder
cd $HBASE_HOME/conf

Then, start hbase
start-hbase.sh


Now, we would like to get the base shell. Use the below command:
hbase shell

That’s it — you have now successfully installed HBase!

Install HappyBase and Dependencies
pip install happybase

Install Apache Thrift (required for happybase):
You can download it from: Apache Thrift.
After installing, make sure thrift is running on your system:bash

start-thrift
Start HBase Thrift Server
$HBASE_HOME/bin/hbase thrift start

Create a table for it  
create 'fraud_test_transactions',
  {NAME => 'transaction_info', VERSIONS => 1},
  {NAME => 'customer_info', VERSIONS => 1},
  {NAME => 'address_info', VERSIONS => 1},
  {NAME => 'merchant_info', VERSIONS => 1}


Run the Python Code

python your_script.py
The Python code is given belowimport csv
import happybase

# Connect to HBase instance
connection = happybase.Connection('localhost', port=9090)
connection.open()

# Get the table
table1 = connection.table('fraud_test_transactions')

# Read the CSV file
with open('fraud test.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        row_key = row['trans_num']  # Using Transaction Number as row key

        # Prepare data to insert into HBase
        table1.put(row_key, {
            'transaction_info:trans_date_trans_time': row['trans_date_trans_time'],
            'transaction_info:cc_num': row['cc_num'],
            'transaction_info:merchant': row['merchant'],
            'transaction_info:category': row['category'],
            'transaction_info:amt': row['amt'],
            'customer_info:first_name': row['first'],
            'customer_info:last_name': row['last'],
            'customer_info:gender': row['gender'],
            'address_info:street': row['street'],
            'address_info:city': row['city'],
            'address_info:state': row['state'],
            'address_info:zip': row['zip'],
            'address_info:lat': row['lat'],
            'address_info:long': row['long'],
            'customer_info:city_pop': row['city_pop'],
            'customer_info:job': row['job'],
            'customer_info:dob': row['dob'],
            'transaction_info:unix_time': row['unix_time'],
            'merchant_info:merch_lat': row['merch_lat'],
            'merchant_info:merch_long': row['merch_long'],
            'transaction_info:is_fraud': row['is_fraud']
        })

# Close the connection
connection.close()Similarly create file for 2nd tableimport csv
import happybase

# Connect to HBase instance
connection = happybase.Connection('localhost', port=9090)
connection.open()

# Get the table
table = connection.table('fraudulent_transactions')

# Read the CSV file
with open('Fraudulent_E-Commerce_Transaction_Data.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        row_key = row['Transaction ID']  # Using Transaction ID as row key

        # Prepare data to insert into HBase
        table.put(row_key, {
            'transaction_info:transaction_amount': row['Transaction Amount'],
            'transaction_info:transaction_date': row['Transaction Date'],
            'transaction_info:transaction_hour': row['Transaction Hour'],
            'customer_info:customer_id': row['Customer ID'],
            'customer_info:customer_age': row['Customer Age'],
            'customer_info:customer_location': row['Customer Location'],
            'product_info:product_category': row['Product Category'],
            'product_info:quantity': row['Quantity'],
            'product_info:payment_method': row['Payment Method'],
            'address_info:shipping_address': row['Shipping Address'],
            'address_info:billing_address': row['Billing Address'],
            'transaction_info:is_fraudulent': row['Is Fraudulent'],
            'transaction_info:account_age_days': row['Account Age Days'],
            'transaction_info:ip_address': row['IP Address'],
            'product_info:device_used': row['Device Used']
        })

# Close the connection
connection.close()

keep the csv and the python in same folder while running

