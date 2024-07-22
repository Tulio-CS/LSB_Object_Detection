### Português

#### Descrição do Projeto

O projeto consiste em um classificador de Língua de Sinais Brasileira (LIBRAS) utilizando a biblioteca MediaPipe para detecção e reconhecimento de gestos em tempo real. A aplicação captura vídeo da câmera do dispositivo, processa os dados utilizando técnicas de visão computacional fornecidas pelo MediaPipe e classifica os gestos identificados por meio de uma rede neural treinada.

#### Funcionalidades Principais

- **Captura de Vídeo em Tempo Real:** Utiliza a câmera do dispositivo para capturar gestos de Língua de Sinais Brasileira.
  
- **Processamento com MediaPipe:** Integração da MediaPipe para detecção e rastreamento de mãos, essencial para identificar gestos.

- **Classificação com Redes Neurais:** Implementação de uma rede neural treinada para classificar os gestos reconhecidos pela MediaPipe.

#### Estrutura do Repositório

```
|- README.md              # Documentação do projeto
|- main.py                # Script principal para execução do aplicativo
|- models/                # Diretório contendo modelos de rede neural treinados
|- utils/                 # Utilitários e funções auxiliares
|- requirements.txt       # Lista de dependências do projeto
|- LICENSE                # Licença de distribuição
```

#### Dependências

- Python 3.x
- MediaPipe
- TensorFlow
- OpenCV

Para instalar as dependências necessárias, execute o seguinte comando:

```bash
pip install -r requirements.txt
```

#### Referências

1. Documentação do MediaPipe: [https://google.github.io/mediapipe/](https://google.github.io/mediapipe/)
2. Documentação do TensorFlow: [https://www.tensorflow.org/api_docs](https://www.tensorflow.org/api_docs)
3. Documentação do OpenCV: [https://docs.opencv.org/](https://docs.opencv.org/)

---

### English

#### Project Description

The project aims to classify Brazilian Sign Language (LIBRAS) using the MediaPipe library for real-time gesture detection and recognition. The application captures video from the device's camera, processes the data using computer vision techniques provided by MediaPipe, and classifies the identified gestures using a trained neural network.

#### Key Features

- **Real-Time Video Capture:** Uses the device camera to capture Brazilian Sign Language gestures.
  
- **MediaPipe Processing:** Integration of MediaPipe for hand detection and tracking, essential for gesture identification.

- **Neural Network Classification:** Implementation of a trained neural network to classify gestures recognized by MediaPipe.

#### Repository Structure

```
|- README.md              # Project documentation
|- main.py                # Main script for running the application
|- models/                # Directory containing trained neural network models
|- utils/                 # Utilities and helper functions
|- requirements.txt       # List of project dependencies
|- LICENSE                # Distribution license
```

#### Dependencies

- Python 3.x
- MediaPipe
- TensorFlow
- OpenCV

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

#### References

1. MediaPipe Documentation: [https://google.github.io/mediapipe/](https://google.github.io/mediapipe/)
2. TensorFlow Documentation: [https://www.tensorflow.org/api_docs](https://www.tensorflow.org/api_docs)
3. OpenCV Documentation: [https://docs.opencv.org/](https://docs.opencv.org/)
