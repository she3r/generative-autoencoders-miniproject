# Autoenkodery + Autoencodery Generatywne.
Autorzy: Piotr Kubacki i Piotr Borycki
Celem dzisiejszych zajęć jest poruszenie następujących tematów:
1. Wprowadzenie i zapoznanie z konceptem AutoEnkodera
2. Wprowadzenie do modeli generatywnych na podstawie VAE
3. Przykład zastosowania autoenkoderów: wykrywanie anomalii
Repozytorium zawiera pliki:
- encoder-01.ipynb -- Wprowadzenie do autoenkoderów: PCA, prosty model Autoenkoder. Zbiory danych: sklearn."UCI ML Wine Data Set dataset", torch.MNIST
- vae-02.ipynb -- Autoenkodery, a modele generatywne: Kullback-Leibler, Variational Autoencoder (VAE), Hipoteza rozmaitości. Zbiory danych: torch.MNIST
- anomaly-03.ipynb -- Zastosowanie autoenkoderów w wykrywaniu anomalii. Interaktywne wprowadzenie do koncepcji wykrywania anomalii (https://anomagram.fastforwardlabs.com/). Zbiory danych: ECG5000 (https://www.timeseriesclassification.com/description.php?Dataset=ECG5000)
- utils.py -- funkcje pomocnocze do notebooków, 
- requirements.txt -- zestaw potrzebnych bibliotek do uruchomienia kodu.

Na zajęciach poruszamy się po plikach według ich numeracji: 01, 02, 03. W każdym notebooku znajdują się zadania (1-4). W każdym zadaniu zostało wyznaczone miejsce na odpowiedź/ rozwiązanie. 

## Konfiguracja środowiska

Stwórz środowisko za pomocą `venv`:
```bash
$ python3.9 -m venv .venv
```
lub z użyciem `conda`:
```bash
$ conda create -n .venv python=3.9
```
Z PyTorch można korzystać przy wsparciu platformy obliczeniowej CUDA, o ile użytkownik ma dostęp do maszyny, która taką platformę wspiera. W przypadku możliwości skorzystania z CUDA, przy instalacji torcha należy kierować się instrukcją instalacji z: https://pytorch.org/

zainstaluj niezbędne biblioteki:
```bash
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

## Źródła:
Źródła zostały nadmienione w notebookach

Dodatkowo, warto zwrócić uwagę na:<br/>
[1] https://www.deeplearningbook.org/contents/autoencoders.html (dostęp na 25.10.2023)<br/>
[2] https://paperswithcode.com/method/vae (dostęp na 25.10.2023)
