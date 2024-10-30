import numpy as np


def detect_head_shoulder(df):
    mask_head_shoulder = ((df.loc[:, "high_roll_max"] > df.loc[:, "High"].shift(1)) & (df.loc[:, "high_roll_max"] > df.loc[:, "High"].shift(-1)) & (df.loc[:, "High"] < df.loc[:, "High"].shift(1)) & (df.loc[:, "High"] < df.loc[:, "High"].shift(-1)))
    mask_inv_head_shoulder = ((df.loc[:, "low_roll_min"] < df.loc[:, "Low"].shift(1)) & (df.loc[:, "low_roll_min"] < df.loc[:, "Low"].shift(-1)) & (df.loc[:, "Low"] > df.loc[:, "Low"].shift(1)) & (df.loc[:, "Low"] > df.loc[:, "Low"].shift(-1)))
    df.loc[:, "head_shoulder_pattern"] = 0
    df.loc[mask_head_shoulder, "head_shoulder_pattern"] = 1
    df.loc[mask_inv_head_shoulder, "head_shoulder_pattern"] = 2
    return ["head_shoulder_pattern"]


def detect_multiple_tops_bottoms(df, window: int):
    df.loc[:, "close_roll_max"] = df.loc[:, "Close"].rolling(window=window).max()
    df.loc[:, "close_roll_min"] = df.loc[:, "Close"].rolling(window=window).min()
    mask_top = (df.loc[:, "high_roll_max"] >= df.loc[:, "High"].shift(1)) & (df.loc[:, "close_roll_max"] < df.loc[:, "Close"].shift(1))
    mask_bottom = (df.loc[:, "low_roll_min"] <= df.loc[:, "Low"].shift(1)) & (df.loc[:, "close_roll_min"] > df.loc[:, "Close"].shift(1))
    df.loc[:, "multiple_top_bottom_pattern"] = 0
    df.loc[mask_top, "multiple_top_bottom_pattern"] = 1
    df.loc[mask_bottom, "multiple_top_bottom_pattern"] = 2
    return ["multiple_top_bottom_pattern"]


def calculate_support_resistance(df, window: int):
    std_dev = 2
    mean_high = df.loc[:, "High"].rolling(window=window).mean()
    std_high = df.loc[:, "High"].rolling(window=window).std()
    mean_low = df.loc[:, "Low"].rolling(window=window).mean()
    std_low = df.loc[:, "Low"].rolling(window=window).std()
    df.loc[:, "support"] = mean_low - std_dev * std_low
    df.loc[:, "resistance"] = mean_high + std_dev * std_high
    return ["support", "resistance"]


def detect_triangle_pattern(df):
    mask_asc = (df.loc[:, "high_roll_max"] >= df.loc[:, "High"].shift(1)) & (df.loc[:, "low_roll_min"] <= df.loc[:, "Low"].shift(1)) & (df.loc[:, "Close"] > df.loc[:, "Close"].shift(1))
    mask_desc = (df.loc[:, "high_roll_max"] <= df.loc[:, "High"].shift(1)) & (df.loc[:, "low_roll_min"] >= df.loc[:, "Low"].shift(1)) & (df.loc[:, "Close"] < df.loc[:, "Close"].shift(1))
    df.loc[:, "triangle_pattern"] = 0
    df.loc[mask_asc, "triangle_pattern"] = 1
    df.loc[mask_desc, "triangle_pattern"] = 2
    return ["triangle_pattern"]


def detect_wedge(df, window: int):
    mask_wedge_up = (df.loc[:, "high_roll_max"] >= df.loc[:, "High"].shift(1)) & (df.loc[:, "low_roll_min"] <= df.loc[:, "Low"].shift(1)) & (df.loc[:, "trend_high"] == 1) & (df.loc[:, "trend_low"] == 1)
    mask_wedge_down = (df.loc[:, "high_roll_max"] <= df.loc[:, "High"].shift(1)) & (df.loc[:, "low_roll_min"] >= df.loc[:, "Low"].shift(1)) & (df.loc[:, "trend_high"] == -1) & (df.loc[:, "trend_low"] == -1)
    df.loc[:, "wedge_pattern"] = 0
    df.loc[mask_wedge_up, "wedge_pattern"] = 1
    df.loc[mask_wedge_down, "wedge_pattern"] = 2
    return ["wedge_pattern"]

def detect_channel(df, window: int):
    channel_range = 0.1
    mask_channel_up = (df.loc[:, "high_roll_max"] >= df.loc[:, "High"].shift(1)) & (df.loc[:, "low_roll_min"] <= df.loc[:, "Low"].shift(1)) & (df.loc[:, "high_roll_max"] - df.loc[:, "low_roll_min"] <= channel_range * (df.loc[:, "high_roll_max"] + df.loc[:, "low_roll_min"])/2) & (df.loc[:, "trend_high"] == 1) & (df.loc[:, "trend_low"] == 1)
    mask_channel_down = (df.loc[:, "high_roll_max"] <= df.loc[:, "High"].shift(1)) & (df.loc[:, "low_roll_min"] >= df.loc[:, "Low"].shift(1)) & (df.loc[:, "high_roll_max"] - df.loc[:, "low_roll_min"] <= channel_range * (df.loc[:, "high_roll_max"] + df.loc[:, "low_roll_min"])/2) & (df.loc[:, "trend_high"] == -1) & (df.loc[:, "trend_low"] == -1)
    df.loc[:, "channel_pattern"] = 0
    df.loc[mask_channel_up, "channel_pattern"] = 1
    df.loc[mask_channel_down, "channel_pattern"] = 2
    return ["channel_pattern"]

def detect_double_top_bottom(df, threshold=0.05):
    mask_double_top = (df.loc[:, "high_roll_max"] >= df.loc[:, "High"].shift(1)) & (df.loc[:, "high_roll_max"] >= df.loc[:, "High"].shift(-1)) & (df.loc[:, "High"] < df.loc[:, "High"].shift(1)) & (df.loc[:, "High"] < df.loc[:, "High"].shift(-1)) & ((df.loc[:, "High"].shift(1) - df.loc[:, "Low"].shift(1)) <= threshold * (df.loc[:, "High"].shift(1) + df.loc[:, "Low"].shift(1))/2) & ((df.loc[:, "High"].shift(-1) - df.loc[:, "Low"].shift(-1)) <= threshold * (df.loc[:, "High"].shift(-1) + df.loc[:, "Low"].shift(-1))/2)
    mask_double_bottom = (df.loc[:, "low_roll_min"] <= df.loc[:, "Low"].shift(1)) & (df.loc[:, "low_roll_min"] <= df.loc[:, "Low"].shift(-1)) & (df.loc[:, "Low"] > df.loc[:, "Low"].shift(1)) & (df.loc[:, "Low"] > df.loc[:, "Low"].shift(-1)) & ((df.loc[:, "High"].shift(1) - df.loc[:, "Low"].shift(1)) <= threshold * (df.loc[:, "High"].shift(1) + df.loc[:, "Low"].shift(1))/2) & ((df.loc[:, "High"].shift(-1) - df.loc[:, "Low"].shift(-1)) <= threshold * (df.loc[:, "High"].shift(-1) + df.loc[:, "Low"].shift(-1))/2)

    df.loc[:, "double_pattern"] = 0
    df.loc[mask_double_top, "double_pattern"] = 1
    df.loc[mask_double_bottom, "double_pattern"] = 2
    return ["double_pattern"]

def detect_trendline(df, window=2):
    df.loc[:, "slope"] = np.nan
    df.loc[:, "intercept"] = np.nan

    for i in range(window, len(df)):
        x = np.array(range(i-window, i))
        y = df.loc[:, "Close"][i-window:i]
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        df.at[df.index[i], "slope"] = m
        df.at[df.index[i], "intercept"] = c

    mask_support = df.loc[:, "slope"] > 0

    mask_resistance = df.loc[:, "slope"] < 0

    df.loc[:, "support_trendline"] = 0.0
    df.loc[:, "resistance_trendline"] = 0.0

    df.loc[mask_support, "support_trendline"] = df.loc[:, "Close"] * df.loc[:, "slope"] + df.loc[:, "intercept"]
    df.loc[mask_resistance, "resistance_trendline"] = df.loc[:, "Close"] * df.loc[:, "slope"] + df.loc[:, "intercept"]
    return ["support_trendline", "resistance_trendline"]

def find_pivots(df):
    high_diffs = df.loc[:, "High"].diff()
    low_diffs = df.loc[:, "Low"].diff()

    higher_high_mask = (high_diffs > 0) & (high_diffs.shift(-1) < 0)
    lower_low_mask = (low_diffs < 0) & (low_diffs.shift(-1) > 0)
    lower_high_mask = (high_diffs < 0) & (high_diffs.shift(-1) > 0)
    higher_low_mask = (low_diffs > 0) & (low_diffs.shift(-1) < 0)

    df.loc[:, "signal"] = 0
    df.loc[higher_high_mask, "signal"] = 1
    df.loc[lower_low_mask, "signal"] = 2
    df.loc[lower_high_mask, "signal"] = 3
    df.loc[higher_low_mask, "signal"] = 4
    return ["signal"]
