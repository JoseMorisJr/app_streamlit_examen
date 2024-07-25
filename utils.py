# Importar las bibliotecas necesarias
from PIL import Image           
from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import torch
import torchvision.transforms as transforms
from torch import autocast

 
def predict_image(image_path):

    # Cargar el modelo y el extractor de características
    model_name = "microsoft/resnet-50"
    extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)

    # Cargar la imagen
    image = Image.open(image_path).convert("RGB")

    # Transformar la imagen a tensor y normalizarla
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=extractor.image_mean, std=extractor.image_std)
    ])
    image = preprocess(image).unsqueeze(0)  # Añadir dimensión de batch

    # Pasar la imagen por el modelo
    with torch.no_grad():
        outputs = model(image)
    logits = outputs.logits

    # Obtener la clase con mayor probabilidad
    predicted_class_idx = logits.argmax(-1).item()
    predicted_class = model.config.id2label[predicted_class_idx]

    return predicted_class


def generacion_imagen(promnt):
    model = "CompVis/stable-diffusion-v1-1"
    image_path_1 = "./image_temp/temporal.png"
    
    scheduler = DDPMScheduler(beta_start=0.001, beta_end=0.005,
                            beta_schedule="scaled_linear", clip_sample = False)
    
    pipe = StableDiffusionPipeline.from_pretrained(model, scheduler=scheduler,
                                        variant="fp16", torch_dtype=torch.float16)
    pipe.to("cuda")
    
    with autocast("cuda"):
        image = pipe(
            promnt, 
            num_inference_steps=30,
            guidance_scale=5,
        ).images[0]
    image.save(image_path_1)
    return image_path_1


    # # Abre la imagen usando PIL
    # image_open = Image.open(image_path_1)

    # # Muestra la imagen en Streamlit
    # st.image(image_open, caption=promnt, use_column_width=True)



