library(keras3)
library(recipes)
library(tidyverse)
library(tidymodels)
library(arrow)
library(viridis)
library(scico)

#load data and append wp
rec_prep <- readRDS("models/wp_recipe.rds")
model <- keras3::load_model("models/nn_wp_model.keras", compile = TRUE)
df <- read_parquet("datasets/clean_pbp.parquet")
baked_full <- bake(rec_prep, new_data = df)
outcomes <- outcome_names(rec_prep)
x_full <- baked_full %>%
  select(-any_of(outcomes)) %>%
  as.matrix()
df$wp <- as.numeric(predict(model, x_full, batch_size = 4096))

#calculate brier score and accuracy
brier <- mean((df$wp - df$home_win)^2, na.rm = TRUE)
accuracy <- mean(ifelse(df$wp > 0.5, 1, 0) == df$home_win, na.rm = TRUE)

#print brier and accuracy
print(paste("Brier Score:", brier))
print(paste("Accuracy:", accuracy))

#viz brier by quarter
p1 <- df |>
  filter(period_number <= 4) |>
  group_by(period_number) |>
  summarise(brier = mean((wp - home_win)^2, na.rm = TRUE)) |>
  ggplot(aes(x = factor(period_number), y = brier)) +
  geom_col(fill = "#2c7fb8", width = 0.7) +
  geom_text(aes(label = sprintf("%.4f", brier)), 
    vjust = -0.5, size = 4, fontface = "bold") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1)), limits = c(0, NA)) +
  labs(
    x = "Quarter",
    y = "Brier Score",
    title = "Model Calibration by Quarter",
    subtitle = "Lower scores indicate better calibration"
  ) +
  theme_bw(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(size = 10, color = "gray30"),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank()
  )
ggsave("plots/brier_by_quarter.png", p1, width = 8, height = 6, dpi = 300)

#viz accuracy by quarter
p2 <- df |>
  filter(period_number <= 4) |>
  group_by(game_id) |>
  mutate(wp_pred = ifelse(wp > .5, 1, 0)) |>
  ungroup() |>
  group_by(period_number) |>
  summarise(accuracy = mean(wp_pred == home_win, na.rm = TRUE)) |>
  ggplot(aes(x = factor(period_number), y = accuracy)) +
  geom_col(fill = "#de8f05", width = 0.7) +
  geom_text(aes(label = scales::percent(accuracy, accuracy = 0.01)), 
    vjust = -0.5, size = 4, fontface = "bold") +
  scale_y_continuous(
    labels = scales::percent_format(),
    expand = expansion(mult = c(0, 0.1)),
    limits = c(0, 1)
  ) +
  labs(
    x = "Quarter",
    y = "Accuracy",
    title = "Prediction Accuracy by Quarter",
    subtitle = "Percentage of correct win/loss predictions"
  ) +
  theme_bw(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(size = 10, color = "gray30"),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank()
  )
ggsave("plots/accuracy_by_quarter.png", p2, width = 8, height = 6, dpi = 300)

#viz data to see average wp given time and score
filtered <- df |>
  filter(!(is.na(home_team_pos))) |>
  mutate(seconds_round = as.integer(round(end_game_seconds_remaining, 0))) |>
  group_by(seconds_round, home_margin, home_team_pos) |>
  summarise(mean_wp = mean(wp))

p3 <- ggplot(filtered, aes(x = seconds_round, y = home_margin, fill = mean_wp)) +
  geom_raster(interpolate = TRUE) +
  scico::scale_fill_scico(
    palette = "vik",
    midpoint = 0.5,
    labels = scales::percent_format(),
    name = "Win\nProbability"
  ) +
  scale_x_reverse(
    expand = c(0,0),
    breaks = seq(0, 2880, 480),
    labels = function(x) paste0(x %/% 60, " min")
  ) +
  scale_y_continuous(expand = c(0,0), breaks = seq(-60, 60, 10)) +
  geom_hline(yintercept = 0, color = "gray20", linewidth = 0.5, alpha = 0.5) +
  labs(
    x = "Time Remaining in Game",
    y = "Home Team Margin",
    title = "Win Probability Throughout Regulation",
    subtitle = "Average home team win probability by score differential and time remaining"
  ) +
  theme_bw(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(size = 10, color = "gray30"),
    legend.position = "right",
    panel.grid = element_blank()
  )
ggsave("plots/wp_heatmap_full.png", p3, width = 12, height = 7, dpi = 300)

p4 <- filtered |>
  filter(seconds_round < 120 & abs(home_margin) <= 9) |>
  ggplot(aes(x = seconds_round, y = home_margin, fill = mean_wp)) +
  geom_raster(interpolate = TRUE) +
  scico::scale_fill_scico(
    palette = "vik",
    midpoint = 0.5,
    labels = scales::percent_format(),
    name = "Win\nProbability"
  ) +
  scale_x_reverse(
    expand = c(0,0),
    breaks = seq(0, 120, 20),
    labels = function(x) paste0(x %/% 60, ":", sprintf("%02d", x %% 60))
  ) +
  scale_y_continuous(
    expand = c(0,0),
    breaks = seq(-9, 9, 3),
    limits = c(-9, 9)
  ) +
  geom_hline(yintercept = 0, color = "gray20", linewidth = 0.5, alpha = 0.5) +
  labs(
    x = "Time Remaining (MM:SS)",
    y = "Home Team Margin",
    title = "Win Probability in Late Game Situations",
    subtitle = "Last 2 minutes of regulation | Games within 9 points"
  ) +
  theme_bw(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(size = 10, color = "gray30"),
    legend.position = "right",
    panel.grid = element_blank()
  )
ggsave("plots/wp_heatmap_clutch.png", p4, width = 12, height = 7, dpi = 300)
