# latex setting
use_latex = True
font_baseline = 24
font_size_general = font_baseline + 2
font_size_axes = font_baseline + 2
font_size_label = font_baseline + 2

# ETH colors
eth_blue = (0.129, 0.361, 0.686)
eth_purple = (0.506757, 0.111486, 0.381757)  # (150, 33, 113) / sum((150, 33, 113))
eth_green = (0.398406, 0.454183, 0.14741)  # (100, 114, 37) / sum((100, 114, 37))
eth_gray = (1 / 3, 1 / 3, 1 / 3)  # (111, 111, 111) / sum(111, 111, 111)
eth_bronze = (142 / 255, 103 / 255, 19 / 255)

# not used
# eth_red = (0.597173, 0.226148, 0.176678)  # (169, 64, 50) / sum(169, 64, 50)
# eth_yellow = (0.941, 0.894, 0.258)

dark_orange = (0.8, 0.4, 0)
dark_red = (0.5, 0, 0)
dark_blue = (0, 0, 0.5)
# \definecolor{darkred}{rgb}{0.5,0,0} % maybe 0.6
# \definecolor{darkorange}{rgb}{0.8, 0.4, 0}

# 0) color scheme of the level sets
# cmap_levelSets = "gray_r"  # "Blues"
cmap_levelSets = "RdBu_r"  # "gray_r"  # "Blues"

# 1) trace plot of iterates in the X and U space
col_opt = "black"
col_feas_tol = eth_gray
# col_feas = "white"
col_feas = "gray"  # dark_orange
alpha_col_feas = 0.7

# col_SwSG = eth_purple  # "purple"
# col_PPM_SwSG = eth_green  # "green"
# col_PPM_ConEx = eth_bronze  # "bronze"
# col_PPM_penalty = eth_blue  # "blue"

col_SwSG = eth_purple  # "purple"
col_PPM_SwSG = eth_purple  # "green"
col_PPM_ConEx = eth_bronze  # "bronze"
col_PPM_ACGD = eth_purple  # "bronze"
col_PPM_penalty = dark_orange  # "blue"


label_SwSG = "SwSG"
label_PPM_SwSG = "IPPM+SwSG"
label_PPM_ConEx = "IPPM+ConEx"
label_PPM_ACGD = "IPPM+ACGD"
label_PPM_penalty = "IPPPM"

axes_label_X = [r"$x_1$", r"$x_2$"]
axes_label_U = [r"$u_1 = c_1(x_1,x_2)$", r"$u_2 = c_2(x_1, x_2)$"]
axes_label_U_not_same_trafo = [r"$u_1$", r"$u_2$"]

marker_start = "o"
marker_last_iterate = "x"
marker_opt = "*"
marker_opt_global = "D"
facecolors_opt_global = "none"

marker_SwSG = "1"
marker_PPM_SwSG = "v"
marker_PPM_ConEx = "*"
marker_PPM_ACGD = "v"
marker_PPM_penalty = "x"

linestyle_SwSG = "dotted"
linestyle_PPM_SwSG = "--"
linestyle_PPM_ConEx = (0, (3, 5, 1, 5, 1, 5))
linestyle_PPM_ACGD = "--"
linestyle_PPM_penalty = "dashdot"
linestyle_feas = "-"

marker_size = 10
scatter_marker_size = 200
alpha_transparency_legend = 1
