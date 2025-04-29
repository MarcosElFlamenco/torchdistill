import torch
from torchdistill.models import resnet


# Load your pre-trained model
model = resnet(20,10,False,False)
state_dict = "resource/ckpt/cifar10/kd/cifar10_resnet20_from_densenet100.pth"
model.load_state_dict(torch.load(state_dict))
model.eval()

# Fuse modules if applicable (e.g., Conv+BN+ReLU)
model_fused = torch.quantization.fuse_modules(model, [['conv', 'bn', 'relu']])

# Set up quantization configuration
model_fused.qconfig = torch.quantization.get_default_qconfig('fbgemm')  # for x86 CPUs
# Use 'qnnpack' for ARM CPUs: torch.quantization.get_default_qconfig('qnnpack')

# Prepare model for quantization
model_prepared = torch.quantization.prepare(model_fused)



# Calibrate with sample data
with torch.no_grad():
    for data, _ in calibration_data_loader:
        model_prepared(data)

# Convert to quantized model
quantized_model = torch.quantization.convert(model_prepared)

# Save quantized model
torch.save(quantized_model.state_dict(), 'quantized_model.pth')