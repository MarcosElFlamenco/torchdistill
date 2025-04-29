include secrets.mk

##LOCAL COMMANDS

STUDENT_MODEL := resnet20
TEACHER_MODEL := densenet100
DATASET := CIFAR5
DISTILL_YAML := $(DATASET)_$(STUDENT_MODEL)_from_$(TEACHER_MODEL).yaml
REMOTE_DISTILL_YAML := remote_distill.yaml
DISTILL_CONFIG := configs/sample/cifar10/kd/$(DISTILL_YAML)
distill:
	python -um examples.torchvision.image_classification \
		--config $(DISTILL_CONFIG) \
		--run_log log/cifar10/kd/resnet20_from_densenet_bc_k12_depth100-hyperparameter_tuning.log \


CE_MODEL := resnet20
CE_DATASET := CIFAR10
CE_YAML := $(CE_DATASET)_$(CE_MODEL)-train.yaml
CE_CONFIG := configs/sample/cifar10/ce/$(CE_YAML)

train_cross_entropy:
	python -um examples.torchvision.image_classification \
		--config $(CE_CONFIG) \
		--run_log log/cifar10/kd/resnet20_from_densenet_bc_k12_depth100-hyperparameter_tuning.log \


#MODEL_TO_VALIDATE := densenet
validate:
	python -um examples.torchvision.image_classification \
		--config $(DISTILL_CONFIG) \
		--run_log log/cifar10/kd/resnet20_from_densenet_bc_k12_depth100-hyperparameter_tuning.log \
		-test_only \
		-student_only


##REMOTE COMMANDS
DISTILL_CLUSTER_NAME := other_distill_cluster
remote_distill:
	export WANDB_API_KEY=$(WANDB_API_KEY) ENV_MODEL_CONFIG=$(DISTILL_CONFIG) && echo $$ENV_MODEL_CONFIG && sky launch -c $(DISTILL_CLUSTER_NAME) --env WANDB_API_KEY --env ENV_MODEL_CONFIG skypilot/$(REMOTE_DISTILL_YAML) -i 10 --down

TRAIN_CLUSTER_NAME := train_cluster
remote_train_ce:
	export WANDB_API_KEY=$(WANDB_API_KEY) ENV_MODEL_CONFIG=$(CE_CONFIG) && echo $$ENV_MODEL_CONFIG && sky launch -c $(TRAIN_CLUSTER_NAME) --env WANDB_API_KEY --env ENV_MODEL_CONFIG skypilot/$(REMOTE_DISTILL_YAML) -i 10 --down 

#depreciated
ssh_retrieve_checkpoint:
	rsync -Pavz $(CLUSTER_NAME):/home/ubuntu/sky_workdir/resource/ckpt/cifar10/kd/cifar10-resnet20_from_densenet_bc_k12_depth100-hyperparameter_tuning.pt /home/oscar/torchdistill/checkpoints

retrieve_distill_checkpoint:
	python -um examples.torchvision.bucket_interactions \
		--config $(DISTILL_CONFIG)

retrieve_ce_checkpoint:
	python -um examples.torchvision.bucket_interactions \
		--config $(CE_CONFIG)

autodown:
	sky autostop distill_cluster -i 5 --down

stat:
	sky status --refresh


##OTHER
]:
	echo "you did it again you mad bastard!!"