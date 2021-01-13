library(tidyverse)
library(tidymodels)
library(ranger)


usethis::use_github()

# Algoritmo de Classificação que irá prever o diagnóstico de Câncer

# Leitura e Analise dos dados(falta colocar os graficos e a exploração de dados)

dados <- read.csv("C:\\Users\\mcaio\\Desktop\\Nova pasta\\data.csv")

dados$id <- NULL

dados$X <- NULL

dados <- dados %>% mutate(diagnosis = as.factor(diagnosis))


# Divisão em base de treino e teste-------------------------------------------------------------

set.seed(1234)

divisao <- initial_split(dados, strata = diagnosis)

base_treinamento <- training(divisao)

base_teste <- testing(divisao)


# BOOTSTRAPS-----------------------------------------------------------------------------------

set.seed(1234)

diagnosis_boot <- bootstraps(base_treinamento)

# RECIPE - TRATAMENTO DOS DADOS------------------------------------------------------------------

diagnostico_receita <-
  recipe(diagnosis ~ ., base_treinamento) %>%
  #step_rm(id, X) %>%  #remover colunas
  step_zv(all_predictors(), -all_outcomes()) %>% #remover colunas com zero variância
  step_normalize(all_numeric(), -all_outcomes())# normalizar os dados(center and scaled)
# step_dummy(all_nominal(), -all_outcomes())#transformar em dummy variaveis nominais(neste caso nao precisa)


# VER COMO FICARAM OS DADOS
dados_recipes <- juice(prep(diagnostico_receita))



# MODELOS DE CLASSIFICAÇÃO ------------------------------------------------------------------

knn_spec <- nearest_neighbor() %>%
  set_engine("kknn") %>%
  set_mode("classification")


log_spec <- logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification")


rf_spec <- rand_forest() %>% 
  set_mode("classification") %>% 
  set_engine("ranger")


# WORKFLOW---------------------------------------------------------------------------------

diagnosis_wf <- workflow() %>% 
  add_formula(diagnosis ~ .)


# FITTANDO OS MODELOS-----------------------------------------------------------------------

knn_fit <- diagnosis_wf %>% 
  add_model(knn_spec) %>% 
  fit_resamples(resamples = diagnosis_boot,
                control = control_resamples(save_pred = TRUE))


log_fit <- diagnosis_wf %>% 
  add_model(log_spec) %>% 
  fit_resamples(resamples = diagnosis_boot,
                control = control_resamples(save_pred = TRUE))


rf_fit <- diagnosis_wf %>% 
  add_model(rf_spec) %>% 
  fit_resamples(resamples = diagnosis_boot,
                control = control_resamples(save_pred = TRUE))


# AVALIANDO A EFICÁCIA DOS MODELOS-------------------------------------------------------------

metrics_models <- bind_rows(log_fit %>% collect_metrics() %>% mutate(model = "log"),
                            knn_fit %>% collect_metrics() %>% mutate(model = "knn"),
                            rf_fit %>% collect_metrics() %>% mutate(model = "rf"))


knn_fit %>% conf_mat_resampled()

log_fit %>% conf_mat_resampled()

rf_fit %>% conf_mat_resampled()


# CURVA ROC------------------------------------------------------------------------------------

bind_rows(knn_fit %>% collect_predictions() %>% 
            roc_curve(truth = diagnosis, .pred_B) %>% 
            mutate(model = "KNN"),
          log_fit %>% collect_predictions() %>% 
            roc_curve(truth = diagnosis, .pred_B) %>% 
            mutate(model = "LOG"),
          rf_fit %>% collect_predictions() %>% 
            roc_curve(truth = diagnosis, .pred_B) %>% 
            mutate(model = "RF")) %>% 
  ggplot(aes(1-specificity, sensitivity, col = model))+
  geom_path(lwd = 1.2)+
  geom_abline(lty = 3)+
  coord_equal()+
  theme_bw()+
  ggtitle("CURVA ROC COMPARATIVA")



# VAMOS SELECIONAR O MODELO KNN------------------------------------------------------------------


knn_fit %>% collect_predictions()


knn_fit %>% collect_predictions() %>% 
  group_by(id) %>% 
  roc_curve(truth = diagnosis, .pred_B) %>% 
  ggplot(aes(1-specificity, sensitivity, color = id))+
  geom_path(lwd = 1.2)+
  geom_abline(lty = 3, size = 1.5)+
  coord_equal()+
  theme_bw()



# SELECIONANDO O MODELO FINAL-----------------------------------------------------------------

diagnosis_final <- diagnosis_wf %>% 
  add_model(knn_spec) %>% 
  last_fit(divisao)

collect_metrics(diagnosis_final)

diagnosis_predictions <- collect_predictions(diagnosis_final)

diagnosis_predictions %>% roc_curve(truth = diagnosis, estimate = .pred_B) %>%
  autoplot()

conf_mat(diagnosis_predictions, truth = diagnosis, estimate = .pred_class)


# ----------------------------------------------------


