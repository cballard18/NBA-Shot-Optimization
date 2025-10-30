library(arrow)
library(tidyverse)
library(recipes)
library(rsample)
library(keras3)

#load and basic cleaning
df <- read_parquet("datasets/clean_pbp.parquet") %>%
  filter(!is.na(home_team_pos))

#game-level split: 20% of games held out for test
set.seed(123)
games <- unique(df$game_id)
test_games <- sample(games, size = ceiling(0.20 * length(games)))
train <- df %>% filter(!game_id %in% test_games)
test <- df %>% filter(game_id %in% test_games)

#oversample end-of-game (<= 60s) in train by 10x
train_os <- train %>%
  mutate(os_wt = if_else(end_game_seconds_remaining <= 60, 10L, 1L)) %>%
  tidyr::uncount(os_wt)

#min-max scaling for predictors (0..1)
rec <- recipe(
  home_win ~ home_margin + home_team_spread +
    end_game_seconds_remaining + home_team_pos,
  data = train_os
) %>%
  step_range(all_predictors(), min = 0, max = 1)

rec_prep <- prep(rec)
train_xy <- bake(rec_prep, new_data = train_os)
test_xy <- bake(rec_prep, new_data = test)

x_train <- as.matrix(select(train_xy, -home_win))
y_train <- as.numeric(train_xy$home_win)
x_test <- as.matrix(select(test_xy, -home_win))
y_test <- as.numeric(test_xy$home_win)

#model: 2 hidden layers (16, 8 relu), sigmoid output, adam 0.001, bce loss
model <- keras_model_sequential() |>
  layer_dense(units = 16, activation = "relu", input_shape = ncol(x_train)) |>
  layer_dense(units = 8, activation = "relu") |>
  layer_dense(units = 1, activation = "sigmoid")

model |> compile(
  optimizer = optimizer_adam(learning_rate = 0.001),
  loss = "binary_crossentropy",
  metrics = list(metric_auc(name = "auc"),
    metric_mean_squared_error(name = "brier"))
)

#early stopping on test-as-validation auc (patience=5, restore best)
cb <- callback_early_stopping(
  monitor = "val_auc", mode = "max", patience = 5, restore_best_weights = TRUE
)

history <- model |> fit(
  x = x_train, y = y_train,
  validation_data = list(x_test, y_test),
  epochs = 100,
  batch_size = 1024,
  verbose = 2,
  callbacks = list(cb)
)

#evaluate
model |> evaluate(x_test, y_test, verbose = 0)

#predict wp (probabilities in [0,1])
wp_test <- as.numeric(model |> predict(x_test))

brier <- mean((wp_test - y_test)^2, na.rm = TRUE)
brier

#threshold at 0.5
pred_cls <- as.integer(wp_test >= 0.5)
acc <- mean(pred_cls == y_test)
acc

#save recipe
saveRDS(rec_prep, "models/wp_recipe.rds")
#save the model
keras3::save_model(model, "models/nn_wp_model.keras")
