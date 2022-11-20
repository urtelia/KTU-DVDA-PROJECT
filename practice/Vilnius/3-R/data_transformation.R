#install.packages("tidyverse")

library(tidyverse)

# Getting the first 1000000 lines.
sample_data <- read_csv("project/1-data/1-sample_data.csv")

# Getting additional 9000000 lines.
additional_data <- read_csv("project/1-data/2-additional_data.csv")

# Merging 1-sample_data.csv with additional_data.csv
data <- merge(sample_data, additional_data, all=TRUE)

select(data, y)

data %>%
  select(y)

additional_features <- read_csv("project/1-data/3-additional_features.csv")

str(additional_features)

# Lets add additional features by ID
joined_data <- inner_join(data, additional_features, by = "id")

#joined_data <- data %>%
#  inner_join(additional_features, by = "id")

write_csv(joined_data, "project/1-data/train_data.csv")  
str(joined_data)
