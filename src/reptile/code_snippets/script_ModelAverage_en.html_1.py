if num_accumulates >= min_average_window and num_accumulates >= min(max_average_window, num_updates * average_window_rate):
    num_accumulates = 0