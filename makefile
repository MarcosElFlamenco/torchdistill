include secrets.mk



STUDENT_MODEL := mobilenet2
TEACHER_MODEL := densenet100
YAML_FILE := $(STUDENT_MODEL)_from_$(TEACHER_MODEL).yaml
REMOTE_YAML := remote_skypilot.yaml
MODEL_CONFIG := configs/sample/cifar10/kd/$(YAML_FILE)

##LOCAL COMMANDS
test_distill:
	python -um examples.torchvision.image_classification \
		--config $(MODEL_CONFIG) \
		--run_log log/cifar10/kd/resnet20_from_densenet_bc_k12_depth100-hyperparameter_tuning.log \

MODEL_TO_VALIDATE := densenet

validate:
	python -um examples.torchvision.image_classification \
		--config $(MODEL_CONFIG) \
		--run_log log/cifar10/kd/resnet20_from_densenet_bc_k12_depth100-hyperparameter_tuning.log \
		-test_only \
		-student_only


##REMOTE COMMANDS
CLUSTER_NAME := distill_cluster
remote_test_distill:
	export WANDB_API_KEY=$(WANDB_API_KEY) ENV_MODEL_CONFIG=$(MODEL_CONFIG) && echo $$ENV_MODEL_CONFIG && sky launch -c $(CLUSTER_NAME) --env WANDB_API_KEY --env ENV_MODEL_CONFIG skypilot/$(REMOTE_YAML) -i 10 --down 

ssh_retrieve_checkpoint:
	rsync -Pavz $(CLUSTER_NAME):/home/ubuntu/sky_workdir/resource/ckpt/cifar10/kd/cifar10-resnet20_from_densenet_bc_k12_depth100-hyperparameter_tuning.pt /home/oscar/torchdistill/checkpoints

aws_retrieve_checkpoint:
	python -um examples.torchvision.bucket_interactions \
		--config $(MODEL_CONFIG)

autodown:
	sky autostop distill_cluster -i 5 --down



stat:
	sky status --refresh

]:
	echo "you did it again you mad bastard!!"