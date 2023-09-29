# Train Kohya-ss style LoRA in Diffusers

### Difference between Kohya-ss LoRA and current Diffusers LoRA (9/29/2023) 

Compared to Kohya-ss LoRA, Diffusers' LoRA is missing the LoRA on the ffn of the BasicTransformerBlock in unet, as well as the LoRA on the convolution layer of proj_in. As for the textencoder, the LoRA in the mlp is missing in Diffusers' version. This training script completes these missing LoRAs, making it identical to Kohya-ss LoRA.

### Requirements

Diffusers version greater than 0.21.2. Because they implemented the Koyha-style LoRA loading, I reused most of their code for my implementation.


### Training script example

I only implement Dreambooth trainig script here. You can run following command to start training:

```bash
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="<your image folder>"
export OUTPUT_DIR="output/<output folder>"

accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=100 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=5000 \
  --validation_prompt="A close potrait of sks dog" \
  --validation_epochs=50 \
  --train_text_encoder \
  --seed="0" \

```

### Inference script example

```py
python inference_text_to_image_lora.py
```

