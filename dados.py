import os
from pathlib import Path

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

class DatasetFolhas(Dataset):
    def __init__(self, raiz, modo='treino'): # modo treino ou teste

        caminho_base = Path(raiz) / modo
        extensoes = {".jpg", ".jpeg", ".png"}
        self.arquivos = sorted(
            str(caminho)
            for caminho in caminho_base.rglob("*")
            if caminho.is_file() and caminho.suffix.lower() in extensoes
        )

        if not self.arquivos:
            raise FileNotFoundError(f"Nenhuma imagem encontrada em {caminho_base}")

        self.transformacao_cor = transforms.Compose([ # 3 canais
            transforms.Resize((256, 256), Image.Resampling.BICUBIC), # artigo usou isso
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.transformacao_cinza = transforms.Compose([
            transforms.Resize((256,256), Image.Resampling.BICUBIC),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])


    def __getitem__(self, index):
        caminho_imagem = self.arquivos[index]
        with Image.open(caminho_imagem) as imagem:
            img_original = imagem.convert('RGB')

        img_cinza = self.transformacao_cinza(img_original)
        img_colorida = self.transformacao_cor(img_original)

        return {"cinza":img_cinza, "colorida":img_colorida}

    def __len__(self):
        return len(self.arquivos)