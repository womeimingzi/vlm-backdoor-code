import torch
import sys
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

model_path = '/data/YBJ/cleansight/models/llava-1.5-7b-hf'
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = LlavaForConditionalGeneration.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16, device_map='cuda:4')

clip_path = getattr(model.config, 'mm_vision_tower', None)
if clip_path:
    from transformers import CLIPVisionModel
    clip_model = CLIPVisionModel.from_pretrained(clip_path, torch_dtype=torch.float16)
    model.vision_tower.load_state_dict(clip_model.state_dict(), strict=False)

model.resize_token_embeddings(len(processor.tokenizer))

sys.path.append('.')
from utils.llava_utils import TrainLLaVACollator, build_qaimage_llava

img = Image.new('RGB', (336, 336), color='white')
q_text = 'Describe this image in a short sentence.'
a_text = 'you have been hacked lol'

col = TrainLLaVACollator(processor, ignore_index=-100)
batch = col([ (q_text, a_text, img, [1]*len(a_text), 0) ])

input_ids = batch['input_ids'].to('cuda:4')
attention_mask = batch['attention_mask'].to('cuda:4')
pixel_values = batch['pixel_values'].to('cuda:4', dtype=torch.float16)
labels = batch['labels'].to('cuda:4')

for name, p in model.named_parameters():
    if 'multi_modal_projector' in name:
        p.requires_grad = True
    else:
        p.requires_grad = False

model.train()
# Try without gradient checkpointing
output = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, labels=labels)
print('Loss (no GC):', output.loss.item())

# Now try with gradient checkpointing
model.enable_input_require_grads()
model.gradient_checkpointing_enable()
output_gc = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, labels=labels)
print('Loss (with GC):', output_gc.loss.item())

output_gc.loss.backward()
for name, p in model.named_parameters():
    if p.requires_grad and p.grad is not None:
        print(name, 'grad norm:', p.grad.norm().item())

