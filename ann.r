# ANN com Keras para Previsao de Pagamentos de Emprestimos Bancarios.
# Ultima modificacao - Wilson - 22/05/2018
#

# Carregando a biblioteca ORE
library(ORE)

# Conectando ao banco de dados.
ore.connect("RQUSER", password="oracle", conn_string="", all=TRUE)

# Carregando tabela para variavel de trabalho local
training_set <- ore.pull(train)
predict_set <- ore.pull(test)

#
# Ajustando "missing values" no training_set
#

# Trocando valores nulos da coluna Gender por "Male".
training_set$Gender[is.na(training_set$Gender)] <- "Male"

# Trocando valores nulos da coluna Married por "Yes".
training_set$Married[is.na(training_set$Married)] <- "Yes"

# Trocando valores nulos da coluna Dependents por "0".
training_set$Dependents[is.na(training_set$Dependents)] <- "0"

# Trocando valores nulos da coluna Self_Employed por "No". 
training_set$Self_Employed[is.na(training_set$Self_Employed)] <- "No"

# Trocando valores nulos da coluna LoanAmount pela media dos valores.
training_set$LoanAmount <- ifelse(is.na(training_set$LoanAmount), 
                                  mean(training_set$LoanAmount, na.rm = TRUE), 
                                  training_set$LoanAmount)

# Trocando valores nulos da coluna Loan_Amount_Term pela mediana dos valores.
training_set$Loan_Amount_Term <- ifelse(is.na(training_set$Loan_Amount_Term),
                                        median(training_set$Loan_Amount_Term,na.rm = TRUE),
                                        training_set$Loan_Amount_Term)

# Trocando valores nulos da coluna Credit_History por 1.
# Credit_history = 0 significa que o cliente tem historico de bom pagador
training_set$Credit_History[is.na(training_set$Credit_History)] <-1

#
# Ajustando "missing values" no predict_set
#

# Trocando valores nulos da coluna Gender por "Male".
predict_set$Gender[is.na(predict_set$Gender)] <- "Male"

# Trocando valores nulos da coluna Married por "Yes".
predict_set$Married[is.na(predict_set$Married)] <- "Yes"

# Trocando valores nulos da coluna Dependents por "0".
predict_set$Dependents[is.na(predict_set$Dependents)] <- "0"

# Trocando valores nulos da coluna Self_Employes por "No". 
predict_set$Self_Employed[is.na(predict_set$Self_Employed)] <- "No"

# Trocando valores nulos da coluna LoanAmount pela media dos valores.
predict_set$LoanAmount <- ifelse(is.na(predict_set$LoanAmount), 
                                 mean(predict_set$LoanAmount, na.rm = TRUE), 
                                 predict_set$LoanAmount)

# Trocando valores nulos da coluna Loan_Amount_Term pela mediana dos valores.
predict_set$Loan_Amount_Term <- ifelse(is.na(predict_set$Loan_Amount_Term),
                                       median(predict_set$Loan_Amount_Term,na.rm = TRUE),
                                       predict_set$Loan_Amount_Term)

# Trocando valores nulos da coluna Credit_History por 1.
# Credit_history = 0 significa que o cliente tem historico de bom pagador
predict_set$Credit_History[is.na(predict_set$Credit_History)] <-1

#
# Codificando as colunas com fatores do training_set (usando biblioteca caret):
#
# install.packages("caret")
# install.packages("DMwR")
library(caret)
library(DMwR)
#
# Loan_Status = Y (ou 0 apos a fatorizacao) significa que o cliente pagou o emprestimo
#
dmy_train <- dummyVars(~ Gender + Married + Dependents + Education + 
                         Self_Employed + Credit_History + Property_Area + Loan_Status,
                       data = training_set)

dmy_train_set <- data.frame(predict(dmy_train, newdata = training_set))

# Removendo as colunas codificadas redundantes para evitar a "Armadilha da Variavel Dummy"
dmy_train_red <- dmy_train_set[c(1,3,5,6,7,9,11,13,14,15,17)]

#
# Removendo as colunas nao utilizadas (redundantes e desnecessarias) do training_set
#
val_train_set <- training_set[c(7,8,9,10)]

# Ajustando a escala e centrando ao redor da media
val_train_centered <- scale(val_train_set, center = TRUE, scale = TRUE)

# Unindo os datasets de treino formando o dataset de aprendizado
learn_set <- cbind(val_train_centered,dmy_train_red)


#
# Executando o mesmo procedimento de fatorizacao e normalizacao para o predict_set
#
dmy_predict <- dummyVars(~ Gender + Married + Dependents + Education + 
                           Self_Employed + Credit_History + Property_Area,
                         data = predict_set)

dmy_predict_set <- data.frame(predict(dmy_predict, newdata = predict_set))
dmy_predict_red <- dmy_predict_set[c(1,3,5,6,7,9,11,13,14,15)]
val_predict_set <- predict_set[c(7,8,9,10)]
val_predict_centered <- scale(val_predict_set, center = TRUE, scale = TRUE)

# Unindo os datasets do dataset de previsao
predict <- cbind(val_predict_centered,dmy_predict_red)

# Dividindo o dataset de aprendizado em dataset de treino e teste
# install.packages("caTools")
library(caTools)

set.seed(123)
split <- sample.split(learn_set$Loan_Status.N, SplitRatio = 0.80)
train_set <- subset(learn_set, split == TRUE)
test_set  <- subset(learn_set, split == FALSE)

x_train_set <- train_set[,-15]
y_train_set <- train_set[,15]

x_test_set <- test_set[,-15]
y_test_set <- test_set[,15]

# Formatando dataset como matriz para uso com Keras
x_train_set <- as.matrix(x_train_set, nrow=c(nrow(x_train_set)), ncol=15)
x_test_set  <- as.matrix(x_test_set, nrow=c(nrow(x_test_set)), ncol=15)

# Modelando a Rede Neural
# Carregando a biblioteca Keras
library(keras)

# Inicializando o modelo
classificador <- keras_model_sequential()

# Adicionando as camadas.
classificador %>%
  layer_dense(units = 14, kernel_initializer="glorot_uniform", input_shape = c(14)) %>% 
  layer_activation('relu') %>%
  layer_dropout(0.2) %>%
  layer_dense(units = 14, kernel_initializer="glorot_uniform") %>% 
  layer_activation('relu') %>%
  layer_dropout(0.2) %>%
  layer_dense(units =  1, kernel_initializer="glorot_uniform") %>% 
  layer_activation('sigmoid')

# Apresentando o resumo do modelo
summary(classificador)

# Compilando o modelo com a funcao de perda crossentropy e otimizador adam
classificador %>% compile(loss = 'binary_crossentropy', 
                          optimizer = optimizer_adam(), 
                          metrics = c('accuracy'))
# Treinando o modelo
history <- classificador %>% fit(x_train_set, y_train_set, epochs = 100, batch_size = 50,  validation_split = 0.25)

# Testando o modelo
taxa_erro_teste <- classificador %>% evaluate(x_test_set, y_test_set)
y_pred <- classificador %>% predict_classes(x_test_set)

# Calculando a Confusion Matrix
cm <- table(y_test_set, y_pred)

# Analises de resultado
Test_TP <- cm[1,1]
Test_TN <- cm[2,2]
Test_FP <- cm[1,2]
Test_FN <- cm[2,1]

TestAccuracy    <- (Test_TP+Test_TN)/(Test_TP+Test_TN+Test_FP+Test_FN)
TestPrecision   <- Test_TP/(Test_TP+Test_FP) 
TestRecall      <- Test_TP/(Test_TP+Test_FN) # Sensivity
TestSpecificity <- Test_TN/(Test_TN+Test_FP)

plot(history)

# Utilizando o modelo com o dataset de previsao.
predict <- as.matrix(predict, nrow=c(nrow(predict)), ncol=15)
predictions <- classificador %>% predict_classes(predict)

histogram(predictions)
histogram(training_set$Loan_Status)

