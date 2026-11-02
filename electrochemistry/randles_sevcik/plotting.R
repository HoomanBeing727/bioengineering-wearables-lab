library(tidyverse)
library(ggsci)

# Run from the repository root after process_data.py has produced
# Data/randles/tidy/<Batch>/summary.csv and slopes.csv.
batch_keep <- c("Bare_Electrode_Tidy", "AuND_Tidy")
search_dir <- Sys.getenv(
  "RANDLES_TIDY_DIR",
  unset = file.path("Data", "randles", "tidy")
)
fig_root <- Sys.getenv(
  "RANDLES_FIG_ROOT",
  unset = file.path("Data", "randles", "figures")
)
out_dir <- file.path(fig_root, "plotting")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

summary_files <- list.files(
  search_dir, pattern = "summary.csv", recursive = TRUE, full.names = TRUE
) %>% 
  set_names() 

df <- map_dfr(summary_files, read_csv, .id = "file_path") %>%
  mutate(Batch = basename(dirname(file_path))) %>%
  filter(Batch %in% batch_keep) %>%
  group_by(Batch, `Scan Rate (V/s)`) %>%
  summarize(Mean_Ip = mean(`Ip (A)`, na.rm = TRUE), .groups = "drop") %>%
  mutate(`Scan Rate (V/s)` = `Scan Rate (V/s)` * 1000) %>%
  rename(scan_rate_mV = `Scan Rate (V/s)`) %>%
  mutate(sqrt_scan_rate = sqrt(scan_rate_mV),
         Mean_Ip = Mean_Ip * 1000) 

slopes_files <- list.files(
  search_dir, pattern = "slopes.csv", recursive = TRUE, full.names = TRUE
) %>% 
  set_names()

slopes_df <- map_dfr(slopes_files, read_csv, .id = "file_path") %>%
  mutate(Batch = basename(dirname(file_path))) %>%
  filter(Batch %in% batch_keep) %>%
  rename(area = `Area (cm^2)`)

p1 <- ggplot(data = df, mapping = aes(x = sqrt_scan_rate, y = Mean_Ip, color = factor(Batch))) +  
  geom_smooth(method = "lm", formula = y ~ x, se = FALSE, 
              linewidth = 0.6, linetype = "dashed") +
  geom_point(size = 2.5) +  
  scale_color_manual(
    values = c("Bare_Electrode_Tidy" = "#4A4A4A", "AuND_Tidy" = "#FF9300"),
    labels = c("Bare_Electrode_Tidy" = "Bare", "AuND_Tidy" = "AuND")
  ) +
  scale_x_continuous(limits = c(2, 16), breaks = c(4, 8, 12, 16), expand = c(0,0)) +
  scale_y_continuous(limits = c(0, 0.10), breaks = seq(0.00, 0.10, by = 0.02), expand = c(0,0)) +
  labs(
    x = expression(paste("[Scan rate (mV/s)]"^"1/2")),
    y = "Current (mA)"
  ) + 
  theme_bw() +
  theme(
    panel.grid.major = element_blank(),
    axis.ticks.length = unit(-0.25, "cm"), 
    panel.grid.minor = element_blank(), 
    panel.border     = element_rect(color = "black", fill = NA, linewidth = 1), 
    axis.title   = element_text(size = 14, color = "black"),
    axis.text    = element_text(size = 12, color = "black"),
    axis.text.x  = element_text(angle = 0, hjust = 0.5), 
    legend.position   = c(0.25, 0.82),
    legend.title      = element_blank(), 
    legend.text       = element_text(size = 11),
    legend.background = element_blank(),
    legend.key        = element_blank(),
    axis.line    = element_blank()
  )

ggsave(file.path(out_dir, "scan_rate.svg"), plot = p1, width = 4.0, height = 4.0, units = "in", dpi = 300)

p2 <- ggplot(data = slopes_df, mapping = aes(x = Batch, y = area)) +  
  stat_summary(fun = "mean", geom = "col", alpha = 0.8, fill='darkorange3', color = "black", linewidth = 0.3) +  
  stat_summary(fun.data = "mean_se", geom = "errorbar", width = 0.2, color = "black", linewidth = 0.8) +  
  labs(title = "Area for each electrode batch",
       x = expression(textstyle(Batch)), 
       y = expression(textstyle(Area)~~(cm^{2}))) +
  scale_x_discrete(
    labels = c("Bare_Electrode_Tidy" = "Bare Electrode", "AuND_Tidy" = "AuND")
  ) +
  theme_bw() +
  theme(
    panel.grid.major = element_blank(),
    axis.ticks.length = unit(-0.25, "cm"), 
    panel.grid.minor = element_blank(), 
    panel.border = element_rect(color = "black", fill = NA, linewidth = 1), 
    plot.title   = element_text(face = "bold", size = 11, hjust = 0.5), 
    axis.title   = element_text(size = 9, color = "black"),
    axis.text    = element_text(size = 8, color = "black"),
    axis.text.x  = element_text(angle = 45, hjust = 1), 
    legend.title = element_text(size = 8, face = "bold"),
    legend.text  = element_text(size = 7),
    axis.line    = element_blank(),
    legend.position = "right"
  )

ggsave(file.path(out_dir, "area.svg"), plot = p2, width = 3.5, height = 2.8, units = "in", dpi = 300)