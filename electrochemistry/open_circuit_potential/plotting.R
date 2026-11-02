library(tidyverse)
library(ggsci)

# ==== Config - change immediately ====
project_folder <- "demo_ocp"
time_window <- c(0,1080) # Total time window in OCP plot
v_window <- c(0,0.3) # Voltage limits in OCP plot
cal_x_lim <- c(1, 50)        # log10 limits in mM for calibration plot
cal_y_lim <- c(0.1, 0.4)   # OCP limits in V for calibration plot 
label_x    <- 4              # label x position in mM (log10 scale)
label_y    <- 0.38      # label y start (change only if batch is true)
label_y_dir <- 0.38    # label y start (change only if direction is true but batch is false)
agg_x_lim <- c(1, 10)        # log10 limits in mM for aggregate calibration plot
agg_y_lim <- c(0.1, 0.20)   # OCP limits in V for aggregate calibration plot 
agg_x <- 20 # calibration plot text position
agg_y <- 0.38 # calibration plot text position
npg_12 <- colorRampPalette(pal_npg("nrc")(10))(12)
# ================

# === Environment - paths are relative to the repository root ===
summary_folder <- paste0(project_folder, "_Final")
full_path <- file.path("Data", project_folder, summary_folder)
summary <- read_csv(file.path(full_path, "ocp_final.csv"))
tidy_long <- read_csv(file.path(full_path, "ocp_tidy_long.csv"))
# ============ 

has_batch <- "Batch" %in% colnames(summary)

# ---- Plot 1: OCP vs Time V-shape ----
tidy_long <- tidy_long %>%
  mutate(Time_offset = `Time/sec` + (Step - 1) * 120)

if (has_batch && "Batch" %in% colnames(tidy_long)) {
  p1 <- ggplot(tidy_long, aes(x = Time_offset, y = `Potential/V`,
                               color = factor(ID), group = interaction(`Batch`, factor(ID)))) +
    facet_wrap(~ `Batch`) +
    labs(x = "Time (s)", y = "Potential (V)", color = "Electrode")
} else {
  p1 <- ggplot(tidy_long, aes(x = Time_offset, y = `Potential/V`,
                               color = factor(ID))) +
    labs(x = "Time (s)", y = "Potential (V)", color = "Electrode")
}

p1 <- p1 +
  geom_line(linewidth = 0.4, alpha = 0.7) +
  theme_minimal() +
  scale_color_manual(values = npg_12) +
  coord_cartesian(xlim = time_window, ylim = v_window) + 
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 1.5),
    axis.text = element_text(size = 12, color = "black"),
    axis.title = element_text(size = 13),
    strip.text = element_text(size = 11, face = "bold")
  )

ggsave(file.path(full_path, "plot_ocp_vs_time.svg"), plot = p1, width = 12, height = 4, units = "in", dpi = 300)

# ---- Plot 2: Calibration Curve ----
slopes <- read_csv(file.path(full_path, "slopes.csv"))
has_direction <- "Direction" %in% colnames(summary)

if (has_batch && has_direction) {
  df_plot <- summary %>%
    group_by(Batch, Direction, ID, Concentration, Units) %>%
    summarise(
      mean_ocp = mean(Mean_OCP, na.rm = TRUE),
      sd_ocp   = sd(Mean_OCP, na.rm = TRUE),
      n        = n(),
      se_ocp   = sd_ocp / sqrt(n),
      .groups  = "drop"
    )

  p2 <- ggplot(df_plot, aes(x = Concentration, y = mean_ocp, color = factor(ID))) +
    geom_smooth(method = "lm", se = FALSE,
                linewidth = 0.6, linetype = "dashed", fullrange = TRUE) +
    geom_errorbar(aes(ymin = mean_ocp - se_ocp, ymax = mean_ocp + se_ocp),
                  width = 4, linewidth = 0.4, alpha = 0.6) +
    geom_point(size = 1.5) +
    facet_grid(Direction ~ Batch) +
    labs(x = "log[Concentration] (mM)", y = "Potential (V)", color = "Electrode") +
    scale_color_manual(values = npg_12)
} else if (has_batch) {
  df_plot <- summary %>%
    group_by(Batch, ID, Concentration, Units) %>%
    summarise(
      mean_ocp = mean(Mean_OCP, na.rm = TRUE),
      sd_ocp   = sd(Mean_OCP, na.rm = TRUE),
      n        = n(),
      se_ocp   = sd_ocp / sqrt(n),
      .groups  = "drop"
    )

  p2 <- ggplot(df_plot, aes(x = Concentration, y = mean_ocp, color = factor(ID))) +
    geom_smooth(method = "lm", se = FALSE,
                linewidth = 0.6, linetype = "dashed", fullrange = TRUE) +
    geom_errorbar(aes(ymin = mean_ocp - se_ocp, ymax = mean_ocp + se_ocp),
                  width = 4, linewidth = 0.4, alpha = 0.6) +
    geom_point(size = 1.5) +
    facet_wrap(~ `Batch`) +
    labs(x = "log[Concentration] (mM)", y = "Potential (V)", color = "Electrode") +
    scale_color_manual(values = npg_12)
} else if (has_direction) {
  df_plot <- summary %>%
    group_by(Direction, ID, Concentration, Units) %>%
    summarise(
      mean_ocp = mean(Mean_OCP, na.rm = TRUE),
      sd_ocp   = sd(Mean_OCP, na.rm = TRUE),
      n        = n(),
      se_ocp   = sd_ocp / sqrt(n),
      .groups  = "drop"
    )

  p2 <- ggplot(df_plot, aes(x = Concentration, y = mean_ocp, color = factor(ID))) +
    geom_smooth(method = "lm", se = FALSE,
                linewidth = 0.6, linetype = "dashed", fullrange = TRUE) +
    geom_errorbar(aes(ymin = mean_ocp - se_ocp, ymax = mean_ocp + se_ocp),
                  width = 4, linewidth = 0.4, alpha = 0.6) +
    geom_point(size = 1.5) +
    facet_wrap(~ Direction) +
    labs(x = "log[Concentration] (mM)", y = "Potential (V)", color = "Electrode") +
    scale_color_manual(values = npg_12)
} else {
  df_plot <- summary %>%
    group_by(ID, Concentration, Units) %>%
    summarise(
      mean_ocp = mean(Mean_OCP, na.rm = TRUE),
      sd_ocp   = sd(Mean_OCP, na.rm = TRUE),
      n        = n(),
      se_ocp   = sd_ocp / sqrt(n),
      .groups  = "drop"
    )

  p2 <- ggplot(df_plot, aes(x = Concentration, y = mean_ocp)) +
    geom_smooth(method = "lm", se = FALSE,
                linewidth = 0.6, linetype = "dashed", color = "black", fullrange = TRUE) +
    geom_errorbar(aes(ymin = mean_ocp - se_ocp, ymax = mean_ocp + se_ocp),
                  width = 4, linewidth = 0.4, color = "black", alpha = 0.6) +
    geom_point(size = 1.5, color = "black") +
    labs(x = "log[Concentration] (mM)", y = "Potential (V)")
}

p2 <- p2 +
    scale_x_log10(limits = cal_x_lim, expand = c(0, 0)) +
  scale_y_continuous(limits = cal_y_lim, expand = expansion(mult = c(0.05, 0.05))) +
  theme_minimal() +
  theme(
    panel.grid.major  = element_blank(),
    axis.ticks.length = unit(-0.2, "cm"),
    panel.grid.minor  = element_blank(),
    panel.border      = element_rect(color = "black", fill = NA, linewidth = 1.5),
    axis.text         = element_text(size = 12, color = "black"),
    axis.text.x       = element_text(angle = 0, hjust = 0.5),
    axis.line         = element_blank(),
    strip.text        = element_text(size = 11, face = "bold")
  )

if (has_batch && has_direction) {
  slope_labels <- df_plot %>%
    filter(Concentration == max(Concentration)) %>%
    slice(1, .by = c(Batch, Direction, ID)) %>%
    left_join(slopes, by = c("Batch", "Direction", "ID")) %>%
    mutate(label = sprintf("%.1f mV/dec", Slope_V_per_decade * 1000)) %>%
    group_by(Batch, Direction) %>%
    mutate(y = label_y_dir - 0.02 * (row_number() - 1)) %>%
    ungroup()
} else if (has_batch) {
  slope_labels <- df_plot %>%
    filter(Concentration == max(Concentration)) %>%
    slice(1, .by = c(Batch, ID)) %>%
    left_join(slopes, by = c("Batch", "ID")) %>%
    mutate(label = sprintf("%.1f mV/dec", Slope_V_per_decade * 1000)) %>%
    group_by(Batch) %>%
    mutate(y = label_y - 0.01 * (row_number() - 1)) %>%
    ungroup()
} else if (has_direction) {
  slope_labels <- df_plot %>%
    filter(Concentration == max(Concentration)) %>%
    slice(1, .by = c(Direction, ID)) %>%
    left_join(slopes, by = c("Direction", "ID")) %>%
    mutate(label = sprintf("%.1f mV/dec", Slope_V_per_decade * 1000)) %>%
    group_by(Direction) %>%
    mutate(y = label_y_dir - 0.02 * (row_number() - 1)) %>%
    ungroup()
} else {
  fit <- lm(mean_ocp ~ log10(Concentration), data = df_plot)
  slope_mV <- coef(fit)[["log10(Concentration)"]] * 1000
  p2 <- p2 + annotate("text", x = label_x, y = label_y,
                      label = sprintf("%.1f mV/dec", slope_mV),
                      family = "mono", fontface = "bold", size = 3)
}

if (exists("slope_labels")) {
  p2 <- p2 +
    geom_text(data = slope_labels,
              aes(x = label_x, y = y, label = label, color = factor(ID)),
              family="mono", fontface="bold", size = 3,
              inherit.aes = FALSE, show.legend = FALSE)
}

ggsave(file.path(full_path, "plot_calibration.svg"), plot = p2, width = 8, height = 6, units = "in", dpi = 300)

# ---- Plot 3: RSD Bar Chart ----
group_cols_rsd <- if (has_batch) c("Concentration", "Units", "Batch") else c("Concentration", "Units")

distinct_cols <- if (has_batch) c("Batch", "ID", "Concentration", "Units") else c("ID", "Concentration", "Units")
rsd <- summary %>%
  distinct(across(all_of(distinct_cols)), .keep_all = TRUE) %>%
  group_by(across(all_of(group_cols_rsd))) %>%
  summarise(
    mean_ocp = mean(Mean_OCP, na.rm = TRUE),
    sd_ocp   = sd(Mean_OCP, na.rm = TRUE),
    rsd      = sd(Mean_OCP, na.rm = TRUE) / mean(Mean_OCP, na.rm = TRUE) * 100,
    .groups  = "drop"
  )

if (has_batch) {
  p3 <- ggplot(rsd, aes(x = factor(Concentration), y = rsd, fill = `Batch`)) +
    geom_col(position = position_dodge(0.9), color = "black", linewidth = 0.6) +
    labs(x = "Concentration (mM)", y = "OCP RSD (%)", fill = "Batch") +
    ggsci::scale_fill_npg()
} else {
  p3 <- ggplot(rsd, aes(x = factor(Concentration), y = rsd)) +
    geom_col(fill = "darkorange3", color = "black", linewidth = 0.6) +
    labs(x = "Concentration (mM)", y = "OCP RSD (%)")
}

p3 <- p3 +
  theme_minimal() +
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 1.5),
    axis.text = element_text(size = 12, color = "black"),
    axis.ticks.length = unit(-0.2, "cm"),
    axis.text.x = element_text(angle = 45, hjust = 1),
    axis.line = element_blank()
  )

if (!has_batch) {
  p3 <- p3 + theme(legend.position = "none")
}

ggsave(file.path(full_path, "plot_rsd.svg"), plot = p3, width = 4, height = 4, units = "in", dpi = 300)

# ---- Plot 4: Aggregate Calibration (black & white) ----
df_agg <- summary %>%
  distinct(ID, Concentration, Units, .keep_all = TRUE) %>%
  group_by(Concentration, Units) %>%
  summarise(
    mean_ocp = mean(Mean_OCP, na.rm = TRUE),
    sd_ocp   = sd(Mean_OCP, na.rm = TRUE),
    n        = n(),
    se_ocp   = sd_ocp / sqrt(n),
    .groups  = "drop"
  )
if (!has_batch) { 
  p4 <- ggplot(df_agg, aes(x = Concentration, y = mean_ocp)) +
    geom_smooth(method = "lm", se = FALSE,
                linewidth = 0.6, linetype = "dashed", color = "black", fullrange=TRUE) +
    geom_errorbar(aes(ymin = mean_ocp - se_ocp, ymax = mean_ocp + se_ocp),
                  width = 0.05, linewidth = 0.5, color = "black") +
    geom_point(size = 2, color = "black") +
    scale_x_log10(limits = agg_x_lim, expand = c(0, 0)) +
    scale_y_continuous(limits = agg_y_lim, expand = expansion(mult = c(0.05, 0.05))) +
    labs(x = "log[Concentration] (mM)", y = "Potential (V)") +
    theme_bw() +
    theme(
      panel.grid.major  = element_blank(),
      axis.ticks.length = unit(-0.2, "cm"),
      panel.grid.minor  = element_blank(),
      panel.border      = element_rect(color = "black", fill = NA, linewidth = 1.5),
      axis.text         = element_text(size = 12, color = "black"),
      axis.text.x       = element_text(angle = 0, hjust = 0.5),
      legend.position   = "none",
      axis.line         = element_blank()
    )
  
  fit_agg <- lm(mean_ocp ~ log10(Concentration), data = df_agg)
  slope_mV_agg <- coef(fit_agg)[["log10(Concentration)"]] * 1000
  p4 <- p4 + annotate("text", x = agg_x, y = agg_y,
                      label = sprintf("%.1f mV/dec", slope_mV_agg),
                      family = "mono", fontface = "bold", size = 4)

  ggsave(file.path(full_path, "plot_aggregate_calibration.svg"), plot = p4, width = 4, height = 4, units = "in", dpi = 300)
}

print("Finished - all OCP plots saved")
