# Classificação automática da condição de vias

Projeto desenvolvido no contexto de um Trabalho de Conclusão de Curso (TCC) para investigar o uso de aprendizado profundo e aprendizado de máquina na classificação automática de imagens de vias.

**Autora:** Hanny Araujo Borges  
**Orientadora:** Elaine Ribeiro de Faria Paiva

> **Objetivo do trabalho**
>
> Avaliar e comparar diferentes estratégias computacionais para identificar a condição de uma via a partir de uma imagem. A automatização desse processo pode apoiar inspeções viárias, reduzir o tempo de levantamento em campo, auxiliar na priorização de manutenções e contribuir para uma gestão mais eficiente da infraestrutura rodoviária.

## Proposta

O trabalho testa três pipelines de classificação binária. A base original possui categorias relacionadas à qualidade da via, que são reorganizadas em duas classes de maior interesse para a avaliação:

| Classe final | Categorias originais |
| --- | --- |
| `good` | `good` e `satisfactory` |
| `poor` | `poor` e `very_poor` |

A comparação permite observar o comportamento de arquiteturas convolucionais utilizadas diretamente para classificação e de uma abordagem híbrida, na qual uma rede neural é usada para extrair representações e o classificador final é um modelo de gradient boosting.

## Arquiteturas avaliadas

### ResNet18: fine-tuning completo

A ResNet18 é inicializada com pesos pré-treinados no ImageNet. A camada final original é substituída por uma camada linear com uma saída, adequada à classificação binária. Todos os parâmetros da rede podem ser atualizados durante o treinamento, caracterizando o **fine-tuning completo**.

### EfficientNet-B0: fine-tuning completo

A EfficientNet-B0 também utiliza pesos pré-treinados no ImageNet. O classificador original é substituído por uma camada linear com uma saída e a rede inteira é otimizada com os dados da tarefa. Essa arquitetura oferece uma alternativa eficiente em número de parâmetros e custo computacional.

### VGG16 + XGBoost: feature extraction

Na terceira estratégia, a VGG16 pré-treinada é utilizada como **extratora de características**. Sua camada classificadora é removida, e os vetores produzidos pela rede são usados como entrada para um classificador XGBoost.

Nesse pipeline:

- a VGG16 permanece em modo de avaliação e seus pesos não são ajustados;
- as características são extraídas para os conjuntos de treino, validação e teste;
- os vetores são normalizados com `StandardScaler`;
- o XGBoost realiza a classificação binária e utiliza o conjunto de validação para acompanhamento do treinamento e early stopping.

## Especificações do experimento

- Imagens convertidas para escala de cinza com três canais.
- Redimensionamento para `224 x 224` pixels.
- Divisão estratificada em `70%` para treino, `15%` para validação e `15%` para teste.
- Execução com as sementes `42`, `123` e `999` para avaliar a estabilidade dos resultados.
- Uso de `BCEWithLogitsLoss` com peso calculado automaticamente a partir da distribuição das classes.
- Otimizador Adam para os pipelines de redes neurais.
- Dez épocas de treinamento nos pipelines de fine-tuning.
- Seleção do melhor modelo neural pelo menor log-loss de validação.
- Avaliação por acurácia, precisão, recall, F1-score e matriz de confusão normalizada em porcentagem.
- Registro de curvas de treinamento, modelos, vetores de características, scaler e métricas em arquivos separados por experimento e semente.
- Uso automático de GPU CUDA quando disponível; caso contrário, o processamento é realizado em CPU.

## Base de dados

A base deve ser obtida no Kaggle:

[Road Damage Classification and Assessment](https://www.kaggle.com/datasets/prudhvignv/road-damage-classification-and-assessment/data)

O download deve ser realizado pelo usuário, respeitando os termos de uso e a licença informados na página da base. Os arquivos de imagem não são incluídos neste repositório.

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

Caso o arquivo compactado do Kaggle apresente uma pasta intermediária ou nomes diferentes, ajuste a estrutura local para que `sih_road_dataset` contenha diretamente uma pasta por classe.

## Como reproduzir

### 1. Clonar o repositório

```bash
git clone <URL_DO_REPOSITORIO>
cd <PASTA_DO_REPOSITORIO>
```

### 2. Criar o ambiente e instalar as dependências

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

Por padrão, o notebook procura a base em `data/sih_road_dataset` e salva os resultados em `outputs/`. Para usar outros diretórios, configure as variáveis de ambiente sem modificar o código.

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

Abra [road_classification.ipynb](road_classification.ipynb) no Jupyter ou no VS Code e execute as células em ordem. A primeira execução pode baixar automaticamente os pesos pré-treinados das arquiteturas a partir do PyTorch.

## Organização dos resultados

Os resultados são gravados em `outputs/`, separados por modelo:

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

Os dados da base, os pesos treinados e os artefatos gerados são ignorados pelo Git para evitar que o repositório fique pesado e para respeitar a distribuição da base original.

## Estrutura do repositório

```text
.
|-- road_classification.ipynb  # notebook principal do experimento
|-- requirements.txt           # dependências Python
|-- README.md                  # documentação e protocolo de reprodução
|-- .gitignore                 # dados e artefatos fora do versionamento
|-- data/                      # base local, não versionada
`-- outputs/                   # resultados locais, não versionados
```

## Observações

Este repositório documenta o experimento computacional do TCC. Os resultados dependem da versão dos pacotes, do hardware disponível, dos pesos pré-treinados e da organização exata da base. Para comparações acadêmicas, recomenda-se registrar o commit do repositório, a versão do ambiente Python, o dispositivo utilizado e as sementes de execução.
