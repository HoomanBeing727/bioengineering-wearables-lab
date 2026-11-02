library(tidyverse)
library(ggsci)

# Run from the repository root. Outputs go under Data/randles/figures/cv/.
search_dir <- Sys.getenv(
  "RANDLES_TIDY_DIR",
  unset = file.path("Data", "randles", "tidy")
)
fig_root <- Sys.getenv(
  "RANDLES_FIG_ROOT",
  unset = file.path("Data", "randles", "figures")
)
out_dir <- file.path(fig_root, "cv")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

summary_files_list <- list.files(
  search_dir, pattern = "summary.csv", recursive = TRUE, full.names = TRUE
)
slopes_files_list  <- list.files(
  search_dir, pattern = "slopes.csv", recursive = TRUE, full.names = TRUE
)

summary_raw <- map_dfr(summary_files_list, read_csv, .id = "file_path")
slopes_raw  <- map_dfr(slopes_files_list, read_csv, .id = "file_path")

summary_raw$file_path <- summary_files_list[as.numeric(summary_raw$file_path)]
slopes_raw$file_path  <- slopes_files_list[as.numeric(slopes_raw$file_path)]

df <- summary_raw %>%
  mutate(Batch = basename(dirname(file_path))) %>%
  group_by(Batch, `Scan Rate (V/s)`) %>%
  summarize(Mean_Ip = mean(`Ip (A)`, na.rm = TRUE), .groups = "drop") %>%
  mutate(sqrt_scan_rate = sqrt(`Scan Rate (V/s)`),
         Mean_Ip = Mean_Ip * 1000)

slopes_df <- slopes_raw %>%
  mutate(Batch = basename(dirname(file_path))) %>%
  select(-file_path, -Electrode_ID) %>% 
  rename(area = `Area (cm^2)`)

p1 <- ggplot(data = df, mapping = aes(x = sqrt_scan_rate, y = Mean_Ip, color = factor(Batch))) +  
  geom_point(size = 1.5, alpha = 0.8) +  
  geom_smooth(method = "lm", se = FALSE, linewidth = 0.8) +
  labs(title = "Peak Current vs. Scan Rate",
       x = expression(textstyle(Scan~Rate)~~(V*s^{-1})^{1/2}), 
       y = expression(Peak~Current~~I[p]~~(mA)), 
       color = "Batch") + 
  ggsci::scale_color_aaas() +
  scale_y_continuous(labels = scales::label_number()) +
  theme_bw() +
  theme(
    panel.grid.major = element_blank(),
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

ggsave(file.path(out_dir, "scan_rate.svg"), plot = p1, width = 3.5, height = 2.8, units = "in", dpi = 300)

p2 <- ggplot(data = slopes_df, mapping = aes(x = Batch, y = area)) + 
  stat_summary(fun = "mean", geom = "col", alpha = 0.8, color = "black", linewidth = 0.3) +  
  stat_summary(fun.data = "mean_se", geom = "errorbar", width = 0.2, color = "black", linewidth = 0.8) +  
  labs(title = "Area for each electrode batch",
       x = "Batch", 
       y = expression(textstyle(Area)~~(cm^{2}))) + 
  theme_bw() +
  theme(
    panel.grid.major = element_blank(),
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