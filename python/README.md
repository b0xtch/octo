deepspeed --num_gpus 2
--num_nodes 2
--master_addr ip-10-0-0-177.us-west-2.compute.internal
--master_port 80
--hostfile hostfile 
inference-test.py 
--model bigscience/bloom-3b