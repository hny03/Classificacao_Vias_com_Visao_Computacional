# Classificacao automatica da condicao de vias

Projeto desenvolvido no contexto de um Trabalho de Conclusao de Curso (TCC) para investigar o uso de aprendizado profundo e aprendizado de maquina na classificacao automatica de imagens de vias.

**Autora:** Hanny Araujo Borges  
**Orientadora:** Elaine Ribeiro de Faria Paiva

> **Objetivo do trabalho**
>
> Avaliar e comparar diferentes estrategias computacionais para identificar a condicao de uma via a partir de uma imagem. A automatizacao desse processo pode apoiar inspeções viarias, reduzir o tempo de levantamento em campo, auxiliar na priorizacao de manutencoes e contribuir para uma gestao mais eficiente da infraestrutura rodoviaria.

## Proposta

O trabalho testa tres pipelines de classificacao binaria. A base original possui categorias relacionadas a qualidade da via, que sao reorganizadas em duas classes de maior interesse para a avaliacao:

| Classe final | Categorias originais |
| --- | --- |
| `good` | `good` e `satisfactory` |
| `poor` | `poor` e `very_poor` |

A comparacao permite observar o comportamento de arquiteturas convolucionais utilizadas diretamente para classificacao e de uma abordagem hibrida, na qual uma rede neural e usada para extrair representacoes e o classificador final e um modelo de gradient boosting.

## Arquiteturas avaliadas

### ResNet18: fine-tuning completo

A ResNet18 e inicializada com pesos pre-treinados no ImageNet. A camada final original e substituida por uma camada linear com uma saida, adequada a classificacao binaria. Todos os parametros da rede podem ser atualizados durante o treinamento, caracterizando o **fine-tuning completo**.

### EfficientNet-B0: fine-tuning completo

A EfficientNet-B0 tambem utiliza pesos pre-treinados no ImageNet. O classificador original e substituido por uma camada linear com uma saida e a rede inteira e otimizada com os dados da tarefa. Essa arquitetura oferece uma alternativa eficiente em numero de parametros e custo computacional.

### VGG16 + XGBoost: feature extraction

Na terceira estrategia, a VGG16 pre-treinada e utilizada como **extratora de caracteristicas**. Sua camada classificadora e removida, e os vetores produzidos pela rede sao usados como entrada para um classificador XGBoost.

Nesse pipeline:

- a VGG16 permanece em modo de avaliacao e seus pesos nao sao ajustados;
- as caracteristicas sao extraidas para os conjuntos de treino, validacao e teste;
- os vetores sao normalizados com `StandardScaler`;
- o XGBoost realiza a classificacao binaria e utiliza o conjunto de validacao para acompanhamento do treinamento e early stopping.

## Especificacoes do experimento

- Imagens convertidas para escala de cinza com tres canais.
- Redimensionamento para `224 x 224` pixels.
- Divisao estratificada em `70%` para treino, `15%` para validacao e `15%` para teste.
- Execucao com as sementes `42`, `123` e `999` para avaliar a estabilidade dos resultados.
- Uso de `BCEWithLogitsLoss` com peso calculado automaticamente a partir da distribuicao das classes.
- Otimizador Adam para os pipelines de redes neurais.
- Dez epocas de treinamento nos pipelines de fine-tuning.
- Selecao do melhor modelo neural pelo menor log-loss de validacao.
- Avaliacao por acuracia, precisao, recall, F1-score e matriz de confusao normalizada em porcentagem.
- Registro de curvas de treinamento, modelos, vetores de caracteristicas, scaler e metricas em arquivos separados por experimento e semente.
- Uso automatico de GPU CUDA quando disponivel; caso contrario, o processamento e realizado em CPU.

## Base de dados

A base deve ser obtida no Kaggle:

[Road Damage Classification and Assessment](https://www.kaggle.com/datasets/prudhvignv/road-damage-classification-and-assessment/data)

O download deve ser realizado pelo usuario, respeitando os termos de uso e a licenca informados na pagina da base. Os arquivos de imagem nao sao incluidos neste repositorio.

Depois de baixar e extrair a base, organize-a no formato esperado pelo `torchvision.datasets.ImageFolder`:

```text
data/
`-- sih_road_dataset/
   |-- good/
   |   `-- imagem_01.jpg
   |-- satisfactory/
   |   `-- imagem_02.jpg
   |-- poor/
   |   `-- imagem_03.jpg
   `-- very_poor/
      `-- imagem_04.jpg
```

Caso o arquivo compactado do Kaggle apresente uma pasta intermediaria ou nomes diferentes, ajuste a estrutura local para que `sih_road_dataset` contenha diretamente uma pasta por classe.

## Como reproduzir

### 1. Clonar o repositorio

```bash
git clone <URL_DO_REPOSITORIO>
cd <PASTA_DO_REPOSITORIO>
```

### 2. Criar o ambiente e instalar as dependencias

```bash
python -m venv .venv
```

No Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

No Linux ou macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configurar a base

Por padrao, o notebook procura a base em `data/sih_road_dataset` e salva os resultados em `outputs/`. Para usar outros diretorios, configure as variaveis de ambiente sem modificar o codigo.

No Windows PowerShell:

```powershell
$env:ROAD_DATASET_DIR = "D:\dados\sih_road_dataset"
$env:ROAD_OUTPUT_DIR = "D:\resultados\road-classification"
```

No Linux ou macOS:

```bash
export ROAD_DATASET_DIR="/dados/sih_road_dataset"
export ROAD_OUTPUT_DIR="/resultados/road-classification"
```

### 4. Executar o notebook

Abra [road_classification.ipynb](road_classification.ipynb) no Jupyter ou no VS Code e execute as celulas em ordem. A primeira execucao pode baixar automaticamente os pesos pre-treinados das arquiteturas a partir do PyTorch.

## Organizacao dos resultados

Os resultados sao gravados em `outputs/`, separados por modelo:

```text
outputs/
|-- ResNet18/
|   |-- models/
|   |-- plots/
|   `-- ResNet18_metrics_<seed>.json
|-- EfficientNet/
|   |-- models/
|   |-- plots/
|   `-- EfficientNet_metrics_<seed>.json
`-- VGG16+XGBoost/
   |-- models/
   |-- plots/
   |-- X_train_<seed>.npy
   |-- X_val_<seed>.npy
   `-- scaler_<seed>.pkl
```

Os dados da base, os pesos treinados e os artefatos gerados sao ignorados pelo Git para evitar que o repositorio fique pesado e para respeitar a distribuicao da base original.

## Estrutura do repositorio

```text
.
|-- road_classification.ipynb  # notebook principal do experimento
|-- requirements.txt           # dependencias Python
|-- README.md                  # documentacao e protocolo de reproducao
|-- .gitignore                 # dados e artefatos fora do versionamento
|-- data/                      # base local, nao versionada
`-- outputs/                   # resultados locais, nao versionados
```

## Observacoes

Este repositorio documenta o experimento computacional do TCC. Os resultados dependem da versao dos pacotes, do hardware disponivel, dos pesos pre-treinados e da organizacao exata da base. Para comparacoes academicas, recomenda-se registrar o commit do repositorio, a versao do ambiente Python, o dispositivo utilizado e as sementes de execucao.
