# Run
deepspeed --num_gpus 2 --num_nodes 2 --master_addr 10.0.0.177 --master_port 80 --hostfile hostfile inference-test.py --model bigscience/bloom-3b

# Install
sudo apt update
sudo apt upgrade
sudo apt-get install libclang-dev

pip install --upgrade pip
pip install deepspeed
pip install -U huggingface_hub

pip install -r requirements.txt

Gen
ssh-keygen -t rsa -b 4096

Copy
cat ~/.ssh/id_rsa.pub

Paste
cat >> ~/.ssh/authorized_keys

Master
`~/.ssh/config`

HostName worker1
User ec2-user

`/etc/hosts`
10.0.0.177 master
10.0.0.24 worker1

Worker
`/etc/hosts`
10.0.0.177 master
10.0.0.24 worker1