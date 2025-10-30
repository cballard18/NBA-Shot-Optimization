library(tidyverse)
library(keras)
library(recipes)
library(tidymodels)
library(arrow)
library(scico)

#load model and recipe
rec_prep <- readRDS("models/wp_recipe.rds")
model <- keras3::load_model("models/nn_wp_model.keras", compile = TRUE)

#create wp predictor function
make_wp_predictor <- function(model, rec_prep, batch_size = 32768) {
  force(model); force(rec_prep)
  function(newdata) {
    stopifnot(is.data.frame(newdata))
    baked <- bake(rec_prep, new_data = newdata)
    preds_mat <- baked %>%
      select(-any_of(outcome_names(rec_prep))) %>%
      as.matrix()
    as.numeric(predict(model, preds_mat, batch_size = batch_size))
  }
}
wp <- make_wp_predictor(model, rec_prep)

#constants
home_team_pos_const <- 0L
home_team_spread_const <- 0

#make/miss and rebound rates
fg3_mean <- 0.358
fg2_mean <- 0.517
fg2_dreb_pct <- 0.670; fg2_oreb_pct <- 0.310
fg3_dreb_pct <- 0.754; fg3_oreb_pct <- 0.243

#time adjustments (seconds)
DT_REB <- 2L
DT_MAKE <- 2L

#base grid
grid <- tidyr::expand_grid(
  home_margin = -16:16,
  end_game_seconds_remaining = 360:0
) %>%
  mutate(
    home_team_pos = home_team_pos_const,
    home_team_spread = home_team_spread_const,
    alt_pos = if_else(home_team_pos == 1L, 0L, 1L)
  )

n <- nrow(grid)

#build all scenarios once (stacked), then one batched predict

#current state at time t
current_df <- transmute(
  grid,
  scenario = "current",
  home_team_pos, home_margin, end_game_seconds_remaining, home_team_spread
)

#defensive rebound branch at time t - DT_REB (possession flips)
fgMiss_df <- transmute(
  grid,
  scenario = "miss_dreb",
  home_team_pos = alt_pos,
  home_margin,
  end_game_seconds_remaining = pmax(0L, end_game_seconds_remaining - DT_REB),
  home_team_spread
)

#offensive rebound branch at time t - DT_REB (same team keeps ball)
keep_df <- transmute(
  grid,
  scenario = "miss_oreb_keep",
  home_team_pos,
  home_margin,
  end_game_seconds_remaining = pmax(0L, end_game_seconds_remaining - DT_REB),
  home_team_spread
)

#made 2 and made 3: deduct DT_MAKE only when > 120s remaining (no stop-clock)
fg2_df <- transmute(
  grid,
  scenario = "plus2",
  home_team_pos = alt_pos,
  home_margin = home_margin + 2,
  end_game_seconds_remaining = if_else(
    end_game_seconds_remaining > 120L,
    pmax(0L, end_game_seconds_remaining - DT_MAKE),
    end_game_seconds_remaining
  ),
  home_team_spread
)
fg3_df <- transmute(
  grid,
  scenario = "plus3",
  home_team_pos = alt_pos,
  home_margin = home_margin + 3,
  end_game_seconds_remaining = if_else(
    end_game_seconds_remaining > 120L,
    pmax(0L, end_game_seconds_remaining - DT_MAKE),
    end_game_seconds_remaining
  ),
  home_team_spread
)

all_scenarios <- bind_rows(current_df, fgMiss_df, keep_df, fg2_df, fg3_df)

#single batched nn predict over everything
all_preds <- wp(all_scenarios)

#reshape back to n x 5 in the same bind order
pred_mat <- matrix(all_preds, nrow = n, ncol = 5, byrow = FALSE)
colnames(pred_mat) <- c("wp_current","wp_miss_dreb","wp_keep","wp_2","wp_3")

wp_current <- pred_mat[, "wp_current"]
wp_miss_dreb <- pred_mat[, "wp_miss_dreb"]
wp_keep <- pred_mat[, "wp_keep"]
wp_2 <- pred_mat[, "wp_2"]
wp_3 <- pred_mat[, "wp_3"]

#normalized miss splits for 2s and 3s
total2 <- fg2_dreb_pct + fg2_oreb_pct
p2_miss_dreb <- (1 - fg2_mean) * (fg2_dreb_pct / total2)
p2_miss_oreb <- (1 - fg2_mean) * (fg2_oreb_pct / total2)

total3 <- fg3_dreb_pct + fg3_oreb_pct
p3_miss_dreb <- (1 - fg3_mean) * (fg3_dreb_pct / total3)
p3_miss_oreb <- (1 - fg3_mean) * (fg3_oreb_pct / total3)

#expected wps (vectorized)
ewp2 <- fg2_mean * wp_2 +
  p2_miss_dreb * wp_miss_dreb +
  p2_miss_oreb * wp_keep

ewp3 <- fg3_mean * wp_3 +
  p3_miss_dreb * wp_miss_dreb +
  p3_miss_oreb * wp_keep

grid_results <- grid %>%
  select(home_margin, end_game_seconds_remaining) %>%
  mutate(ewp3_net = ewp3 - ewp2)

write_parquet(grid_results, "datasets/eWP3_results.parquet")

#overall plot of results
p1 <- grid_results |>
  ggplot(aes(end_game_seconds_remaining, home_margin, fill = ewp3_net)) +
  geom_raster(interpolate = TRUE) +
  scale_x_reverse(expand = c(0,0)) +
  scale_y_continuous(expand = c(0,0)) +
  scico::scale_fill_scico(palette = "vik", midpoint = 0, oob = scales::squish, label = percent) +
  theme_bw() +
  labs(x = "Time Remaining (Seconds)", y = "Home Margin",
    title = "Expected WP Added by Shooting a 3",
    fill = "FG3 eWPA", subtitle = "Home margin within 20, Q4 only")
ggsave("plots/ewp3_heatmap.png", p1, width = 10, height = 8, dpi = 300)

#class results plot
p2 <- grid_results |>
  mutate(optimal_shot = case_when(
    ewp3_net > 0.01 ~ "3-Point Shot",
    ewp3_net < -0.01 ~ "2-Point Shot",
    TRUE ~ NA
  )) |>
  filter(!(is.na(optimal_shot))) |>
  ggplot(aes(end_game_seconds_remaining, home_margin, fill = optimal_shot)) +
  geom_tile(color = "white", linewidth = 0.25) +
  scale_fill_manual(
    values = c("3-Point Shot" = "#1f77b4", "2-Point Shot" = "#ff7f0e"),
    name = "Optimal Shot"
  ) +
  scale_x_reverse(
    expand = c(0,0),
    breaks = seq(0, 360, 15),
    limits = c(395, -35),
    labels = function(x) {
      ifelse(x %% 60 == 0 & x >= 0 & x <= 360,
        paste0(x %/% 60, ":", sprintf("%02d", x %% 60)),
        "")
    }
  ) +
  scale_y_continuous(
    expand = c(0,0),
    breaks = seq(-16, 16, 1),
    limits = c(-16, 16)
  ) +
  geom_hline(yintercept = 0, color = "gray30", linewidth = 0.8, linetype = "dashed") +
  annotate("text", x = 170, y = -3, label = "Zone 1", 
    size = 3.5, fontface = "bold", color = "#ff7f0e", hjust = 1) +
  annotate("segment", x = 160, xend = 110, y = -3, yend = -3,
    arrow = arrow(length = unit(0.2, "cm"), type = "closed"),
    color = "#ff7f0e", linewidth = 0.8) +
  annotate("text", x = 220, y = 2, label = "Zone 2", 
    size = 3.5, fontface = "bold", color = "#1f77b4", hjust = 1) +
  annotate("segment", x = 210, xend = 150, y = 2, yend = 2,
    arrow = arrow(length = unit(0.2, "cm"), type = "closed"),
    color = "#1f77b4", linewidth = 0.8) +
  annotate("text", x = 30, y = 6.5, label = "Zone 3", 
    size = 3.5, fontface = "bold", color = "#ff7f0e", hjust = 0) +
  annotate("segment", x = 23, xend = 23, y = 6, yend = 2,
    arrow = arrow(length = unit(0.2, "cm"), type = "closed"),
    color = "#ff7f0e", linewidth = 0.8) +
  annotate("text", x = 30, y = -9.5, label = "Zone 4", 
    size = 3.5, fontface = "bold", color = "#1f77b4", hjust = 0) +
  annotate("segment", x = 23, xend = 23, y = -9, yend = -7,
    arrow = arrow(length = unit(0.2, "cm"), type = "closed"),
    color = "#1f77b4", linewidth = 0.8) +
  theme_bw(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(size = 10, color = "gray30"),
    legend.position = "bottom",
    legend.title = element_text(face = "bold"),
    panel.grid.major = element_line(color = "gray80", linewidth = 0.3),
    panel.grid.minor = element_blank(),
    plot.margin = margin(5.5, 60, 5.5, 60, "pt")
  ) +
  coord_cartesian(clip = "off") +
  labs(
    x = "Time Remaining (MM:SS)",
    y = "Home Team Margin",
    title = "Optimal Shot Selection by Game Situation",
    subtitle = "Based on Expected Win Probability Added (eWPA > 1% threshold) | Last 6 Minutes of Game"
  )
ggsave("plots/shot_type_recommendation.png", p2, width = 12, height = 7, dpi = 300)
