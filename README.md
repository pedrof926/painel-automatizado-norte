# Painel de Monitoramento de Calor Extremo - Região Norte do Brasil

Este painel interativo exibe a previsão de eventos de calor extremo com base no índice EHF (Excess Heat Factor), utilizando a previsão do INMET e os percentis históricos calculados a partir da base BR-DWGD.

## Funcionalidades

- Mapa interativo por município da Região Norte com classificação do EHF (Normal, Severo, Extremo)
- Gráfico com temperaturas máxima, mínima e média dos próximos 5 dias ao clicar no município
- Atualização automática com os dados mais recentes do INMET
- Armazenamento automático dos últimos 30 dias de Tmédia para cálculo do EHF

## Como rodar localmente

1. Clone o repositório:
```bash
git clone https://github.com/seuusuario/painel-ehf-norte.git
cd painel-ehf-norte
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Execute o painel:
```bash
python app.py
```

---

Os arquivos `historico_tmédia/` e `ehf_previsao/` serão gerados automaticamente durante a execução do script `atualizar_ehf_diario.py`.