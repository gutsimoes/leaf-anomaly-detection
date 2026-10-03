import torch.nn as nn
import torch

def inicializar_pesos(m):
    # pesos "aleatorios" q seguem distribuicao
    nome_classe = m.__class__.__name__
    if nome_classe.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif nome_classe.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class BlocoDescida(nn.Module):
    #parte do encoder, reduzir tamanho + extrair informacao
    def __init__(self, canais_entrada, canais_saida, normalizar=True, dropout=0.0):
        super(BlocoDescida, self).__init__()
        camadas = [nn.Conv2d(canais_entrada, canais_saida, kernel_size=4, stride=2, padding=1, bias=False)]

        if normalizar:
            camadas.append(nn.InstanceNorm2d(canais_saida))
        camadas.append(nn.LeakyReLU(0.2))

        if dropout :
            camadas.append(nn.Dropout(dropout))

        self.modelo = nn.Sequential(*camadas)

    def forward(self, x):
        return self.modelo(x)

class BlocoSubida(nn.Module):
    #parte do decoder
    def __init__(self, canais_entrada, canais_saida, dropout=0.0):
        super(BlocoSubida, self).__init__()
        # ConvTranspose2d p upsampling
        camadas = [
            nn.ConvTranspose2d(canais_entrada, canais_saida, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(canais_saida),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            camadas.append(nn.Dropout(dropout))

        self.modelo = nn.Sequential(*camadas)

    def forward(self, x, skip_input):
        x = self.modelo(x)
        x = torch.cat((x, skip_input), 1)
        return x

class GeradorUNet(nn.Module):
    # reconstrói as cores (recebe 1 canal, sai 3)
    def __init__(self, canais_entrada=1, canais_saida=3):
        super(GeradorUNet, self).__init__()

        #descdida
        self.descida1 = BlocoDescida(canais_entrada, 64, normalizar=False)
        self.descida2 = BlocoDescida(64, 128)
        self.descida3 = BlocoDescida(128, 256)
        self.descida4 = BlocoDescida(256, 512, dropout=0.5)
        self.descida5 = BlocoDescida(512, 512, dropout=0.5)
        self.descida6 = BlocoDescida(512, 512, dropout=0.5)
        self.descida7 = BlocoDescida(512, 512, dropout=0.5)
        self.descida8 = BlocoDescida(512, 512, normalizar=False, dropout=0.5) # O fundo do "U"

        #subida
        self.subida1 = BlocoSubida(512, 512, dropout=0.5)
        self.subida2 = BlocoSubida(1024, 512, dropout=0.5)
        self.subida3 = BlocoSubida(1024, 512, dropout=0.5)
        self.subida4 = BlocoSubida(1024, 512, dropout=0.5)
        self.subida5 = BlocoSubida(1024, 256)
        self.subida6 = BlocoSubida(512, 128)
        self.subida7 = BlocoSubida(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, canais_saida, kernel_size=4, padding=1),
            nn.Tanh(), # -1 <= x <= 1
        )

    def forward(self, x):
        # guarda pra usar na subida
        d1 = self.descida1(x)
        d2 = self.descida2(d1)
        d3 = self.descida3(d2)
        d4 = self.descida4(d3)
        d5 = self.descida5(d4)
        d6 = self.descida6(d5)
        d7 = self.descida7(d6)
        d8 = self.descida8(d7)

        # conectando
        u1 = self.subida1(d8, d7)
        u2 = self.subida2(u1, d6)
        u3 = self.subida3(u2, d5)
        u4 = self.subida4(u3, d4)
        u5 = self.subida5(u4, d3)
        u6 = self.subida6(u5, d2)
        u7 = self.subida7(u6, d1)

        return self.final(u7)

class Discriminador(nn.Module):
    #julga se o par cinza + colorido é real ou falso (gerado), exige textura nítida pq avalia patches

    def __init__(self, canais_entrada=1, canais_saida=3):
        super(Discriminador, self).__init__()

        canais_totais = canais_entrada + canais_saida

        def bloco_discriminador(entrada, saida, normalizar=True):
            camadas = [nn.Conv2d(entrada, saida, kernel_size=4, stride=2, padding=1)]
            if normalizar:
                camadas.append(nn.InstanceNorm2d(saida))
            camadas.append(nn.LeakyReLU(0.2, inplace=True))
            return camadas

        self.modelo = nn.Sequential(
            *bloco_discriminador(canais_totais, 64, normalizar=False),
            *bloco_discriminador(64, 128),
            *bloco_discriminador(128, 256),
            *bloco_discriminador(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),

            nn.Conv2d(512, 1, kernel_size=4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        img_input = torch.cat((img_A, img_B), 1)
        return self.modelo(img_input)