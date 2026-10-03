# Diagnóstico de anomalias em folhas

Projeto acadêmico concluído de detecção de anomalias em folhas por reconstrução de cores. Este repositório é publicado exclusivamente para arquivamento, consulta e preservação do trabalho; não está em desenvolvimento ativo e não há atualizações planejadas.

## Metodo

O gerador U-Net aprende a colorir imagens em tons de cinza de folhas saudáveis. A diferença entre a imagem original e a reconstruída é medida com CIEDE2000; valores maiores podem indicar regiões anômalas. O discriminador PatchGAN participa do treinamento, e o notebook também gera uma visualização Grad-CAM.

Implementacao baseada no trabalho de Ryoya Katafuchi e Terumasa Tokunaga, *Image-based plant disease diagnosis with unsupervised anomaly detection based on reconstructability of colors* (2020), [arXiv:2011.14306](https://arxiv.org/abs/2011.14306).

## Arquivos

- `Projeto.ipynb`: treinamento, avaliacao, metricas e visualizacoes.
- `dados.py`: carregamento e transformacao das imagens.
- `models.py`: gerador U-Net, discriminador PatchGAN e inicializacao.

## Ambiente

Use Python 3.10 ou superior. Instale PyTorch e Torchvision de acordo com o sistema e o hardware disponíveis; para AMD/ROCm, siga as instruções oficiais do PyTorch. Em seguida, instale as demais dependências:

```bash
python -m pip install -r requirements.txt
```

Abra `Projeto.ipynb` no VS Code ou Jupyter, selecione esse ambiente como kernel e execute as células em ordem. O notebook usa `data/` relativo ao diretório de trabalho atual. O treinamento está configurado para 100 épocas; para uma verificação curta, altere temporariamente `epochs` e depois restaure o valor.

## Dados

As imagens não estão incluídas neste repositório. Obtenha-as da fonte autorizada e organize-as nesta estrutura:

```text
data/
  treino/
    saudavel/       # 50 imagens saudáveis
  teste/
    saudavel/       # 50 imagens saudáveis
    doente/         # 100 imagens doentes
```

O `DatasetFolhas` procura imagens JPG, JPEG e PNG recursivamente dentro de `treino/` ou `teste/`. Para a avaliação, os caminhos das imagens de teste precisam identificar a classe com `saudavel`/`healthy` ou `doente`/`disease`.

Não publique imagens fornecidas para fins acadêmicos sem permissão dos titulares.

## Resultados registrados

Uma execução registrada com 100 épocas apresentou AUC ROC de 0,9578, precisão de 0,9468, recall de 0,8900 e F1 de 0,9175. O threshold foi escolhido pelo critério de Youden no próprio conjunto de teste; por isso, precisão, recall e F1 são resultados exploratórios e podem estar otimistas. Eles não devem ser interpretados como uma estimativa independente de desempenho.

O score de anomalia é a soma das diferenças CIEDE2000 por pixel e depende do tamanho/região da folha. AUC e métricas também dependem da organização e dos rótulos corretos dos dados.

## Artefatos

Dados, checkpoints, pesos finais e imagens intermediárias são ignorados pelo Git. Os gráficos e métricas já salvos como outputs em `Projeto.ipynb` permanecem versionados como registro da execução arquivada.