args <- commandArgs(TRUE)

file_name <- args[1]
index <- args[2]

data_name <- load(file_name)

export_to_gnn(eval(parse(text = data_name)), index)