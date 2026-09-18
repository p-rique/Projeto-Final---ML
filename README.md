# Projeto Final - ML

## Respondendo às perguntas para a conclusão:

1 - Qual era o objetivo do projeto?

> Treinar um modelo de Machine Learning para o estudo de um banco de dados de vinho. Com o objetivo de analisar características químicas para prever corretamente e classificar a qual categoria o vinho pertence.

2 - Qual dataset foi escolhido?

> O dataset Wine da biblioteca scikit-learn.

3 - Quais cuidados foram necessários na preparação?

> Inspecionar os dados, analisando as classes, os tipos dos dados e validando que não existiam valores ausentes nem duplicatas no dataset. Além da separação dos dados features (X) do target (y) fazendo a divisão de 80% treino e de 20% teste.

4 - Por que o modelo foi escolhido?

> Após fazer o teste com o modelo de Decision Tree percebi que a acurácia estava próxima mas não chegava ao nível esperado, por isso testei o Random Forest Classifier, que alcançou os níveis esperados, e por isso foi o modelo escolhido.

5 - Por que essas métricas foram utilizadas?

> Acurácia: Para verificar a proporção total de acertos do modelo;
> Validação Cruzada: Para validar a acurácia do modelo, garantindo que não estava overfitting;
> Precisão: Para verificar a proporção de previsões de uma classe que o modelo fez estavam de fato corretas.

6 - O resultado foi satisfatório? Justifique com números.

> Sim. Alcançamos uma precisão de 1.0 e uma acúracia média de 0.972.

7 - Existem sinais de overfitting ou underfitting?

> Não, ao analisar a acurácia tivemos um resultado igual a 1.0, o que levantou a suspeita, porém ao verificar com uma validação cruzada, obtivemos uma média de 0.972, provando que o modelo está funcional e não está sofrendo de overfitting, mantendo o nível alto esperado.

8 - O que poderia ser melhorado?

> Poderia testar uma normalização dos dados com o StandardtScaler