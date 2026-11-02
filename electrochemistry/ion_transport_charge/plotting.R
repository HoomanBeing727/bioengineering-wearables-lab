library(tidyverse)

# Run this script from the repository root, after main.py has produced
# Data/demo_ion/demo_ion_Final/*.csv.
project_folder <- "demo_ion"
data_folder <- paste0(project_folder, "_Final")
data_dir <- file.path("Data", project_folder, data_folder)

summary_files <- list.files(
  data_dir,
  pattern = "Tidy_ID_\\d+\\.\\d+\\.csv",
  full.names = TRUE
)

df <- map_dfr(summary_files, read_csv, .id = "file_path") %>%
  filter(`Time/sec` == 120) %>%
  select(-file_path)

df_summary <- df %>%
  group_by(Concentration) %>%
  summarize(
    mean_charge = mean(`Charge/C`, na.rm = TRUE),
    sd_charge   = sd(`Charge/C`, na.rm = TRUE),
    n           = n(),
    se_charge   = sd_charge / sqrt(n),
    .groups     = "drop"
  )

# Calibration curve
p1 <- ggplot(data = df_summary, mapping = aes(x = Concentration, y = mean_charge)) +
  geom_smooth(method = "lm", formula = y ~ x - 1, se = FALSE,
              linewidth = 0.6, linetype = "dashed", color = "black") +
  geom_errorbar(aes(ymin = mean_charge - se_charge, ymax = mean_charge + se_charge),
                linewidth = 0.6, color = "black") +
  geom_point(size = 2, color = "black") +
  scale_x_continuous(limits = c(0, NA), expand = expansion(mult = c(0, 0.05))) +
  scale_y_continuous(limits = c(0, NA), expand = expansion(mult = c(0, 0.05))) +
  labs(x = "Concentration (mM)", y = "Charge (C)") +
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

ggsave(file.path(data_dir, "plot1.png"), plot = p1, width = 4.0, height = 4.0,
       units = "in", dpi = 300)

print("Finished")
