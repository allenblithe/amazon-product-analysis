# -*- coding: utf-8 -*-
"""
============================================================
Amazon 商品运营分析项目（第二个项目）
============================================================
数据：1000 个 Amazon 商品的真实抓取数据（含价格、评分、评论、
      BSR 排名、图片数、卖家数等 55 个字段）
来源：GitHub bright-cn/Amazon-dataset-samples

分析目标（对应跨境电商运营岗的真实问题）：
  1. 市场概览：品类 / 价格 / 评分分布
  2. 什么因素影响商品排名（BSR）？
  3. 爆款商品有什么共同特征？
  4. 给运营的可执行优化建议

作者：钱陶勇 ｜ 2027届统计学
============================================================
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

sns.set_style("whitegrid")
# 字体设置必须在 sns.set_style 之后（否则会被覆盖）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110

DATA = "data/amazon_products.csv"
FIG = "figures"
OUT = "output"


# ============================================================
# 第 1 步：数据清洗
# ============================================================
def clean_price(v):
    """价格字段：去引号 + 转 float（处理科学计数法）"""
    if pd.isna(v):
        return np.nan
    s = str(v).strip().strip('"').strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def clean_discount(v):
    """折扣字段：'-32%' → -32.0"""
    if pd.isna(v):
        return np.nan
    s = str(v).replace("%", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def top_category(s):
    """类目字段：JSON 数组 → 一级类目"""
    if pd.isna(s):
        return None
    try:
        arr = json.loads(s)
        return arr[0] if arr else None
    except (json.JSONDecodeError, TypeError):
        return None


def load_and_clean(path=DATA):
    print("=" * 64)
    print("第 1 步：数据加载与清洗")
    print("=" * 64)
    df = pd.read_csv(path)
    print(f"原始数据：{df.shape[0]:,} 行 × {df.shape[1]} 列")

    # 价格、折扣、类目
    df["price"] = df["final_price"].apply(clean_price)
    df["initial_price_num"] = df["initial_price"].apply(clean_price)
    df["discount_pct"] = df["discount"].apply(clean_discount)
    df["main_category"] = df["categories"].apply(top_category)

    # 关键字段的缺失情况
    print("\n关键字段缺失：")
    for c in ["price", "rating", "reviews_count", "root_bs_rank"]:
        n = df[c].isna().sum()
        print(f"  {c:<16} 缺失 {n:>4} 条 ({n/len(df):.1%})")

    # 保留分析所需字段，去掉缺 BSR 的
    before = len(df)
    df = df.dropna(subset=["root_bs_rank"]).copy()
    print(f"\n去掉无 BSR 的商品：{before} → {len(df)} 行")

    # BSR 分档（越小越好）
    df["bsr_level"] = pd.cut(df["root_bs_rank"],
                             bins=[0, 5000, 20000, 80000, np.inf],
                             labels=["Top 5000", "5000-2万", "2万-8万", "8万+"])
    print("\nBSR 分档分布：")
    print(df["bsr_level"].value_counts().sort_index().to_string())
    print(f"\n价格范围：${df['price'].min():.2f} ~ ${df['price'].max():.2f}")
    print(f"评分范围：{df['rating'].min():.1f} ~ {df['rating'].max():.1f}")
    print(f"评论数范围：{df['reviews_count'].min():.0f} ~ {df['reviews_count'].max():.0f}")
    return df


# ============================================================
# 第 2 步：市场概览
# ============================================================
def market_overview(df):
    print("\n" + "=" * 64)
    print("第 2 步：市场概览")
    print("=" * 64)

    # 类目分布
    cat = df["main_category"].value_counts().head(10)
    print("一级类目 Top 10：")
    print(cat.to_string())

    # 价格带
    df["price_band"] = pd.cut(df["price"],
                              bins=[0, 15, 25, 50, 100, np.inf],
                              labels=["<$15", "$15-25", "$25-50", "$50-100", "$100+"])
    print("\n价格带分布：")
    print(df["price_band"].value_counts().sort_index().to_string())

    # 图 1：类目 + 价格分布
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].barh(cat.index[::-1], cat.values[::-1], color="#1a5fb4")
    axes[0].set_title("一级类目商品数 Top 10", fontsize=12)
    axes[0].set_xlabel("商品数")
    axes[1].hist(df["price"].dropna(), bins=40, color="#e07b39", edgecolor="white")
    axes[1].set_title("商品价格分布", fontsize=12)
    axes[1].set_xlabel("价格 ($)"); axes[1].set_ylabel("频数")
    plt.tight_layout()
    plt.savefig(f"{FIG}/01_market_overview.png", bbox_inches="tight")
    plt.close()

    print(f"\n结论：类目集中在「{cat.index[0]}」（{cat.values[0]} 个，占 {cat.values[0]/len(df):.1%}）；"
          f"价格中位数 ${df['price'].median():.2f}")
    return df


# ============================================================
# 第 3 步：什么因素影响商品排名（BSR）？
# ============================================================
def bsr_factors(df):
    print("\n" + "=" * 64)
    print("第 3 步：什么因素影响商品排名（BSR）？")
    print("=" * 64)

    # 相关性（Spearman：对排名这种非正态分布更稳健）
    cols = ["price", "rating", "reviews_count", "images_count",
            "video_count", "number_of_sellers", "answered_questions"]
    corr = {}
    for c in cols:
        sub = df[[c, "root_bs_rank"]].dropna()
        if len(sub) > 10:
            corr[c] = sub[c].corr(sub["root_bs_rank"], method="spearman")
    corr_s = pd.Series(corr).sort_values()
    print("与 BSR 的 Spearman 相关系数（负值 = 该指标越大、排名越靠前）：")
    print(corr_s.round(3).to_string())
    print("\n> 注意：BSR 是'越小越好'，所以负相关意味着'指标越大→排名越靠前'")

    # 分箱对比：评论数、评分、图片数
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # 评论数分箱
    df["review_band"] = pd.cut(df["reviews_count"],
                               bins=[0, 50, 200, 1000, 5000, np.inf],
                               labels=["<50", "50-200", "200-1k", "1k-5k", "5k+"])
    g1 = df.groupby("review_band", observed=True)["root_bs_rank"].median()
    axes[0].bar(g1.index.astype(str), g1.values, color="#1a5fb4")
    axes[0].set_title("评论数 vs BSR 中位数（越低越好）", fontsize=11)
    axes[0].set_xlabel("评论数"); axes[0].set_ylabel("BSR 中位数")
    axes[0].tick_params(axis="x", rotation=20)

    # 评分分箱
    df["rating_band"] = pd.cut(df["rating"],
                               bins=[0, 4.0, 4.3, 4.5, 4.7, 5.0],
                               labels=["<4.0", "4.0-4.3", "4.3-4.5", "4.5-4.7", "4.7+"])
    g2 = df.groupby("rating_band", observed=True)["root_bs_rank"].median()
    axes[1].bar(g2.index.astype(str), g2.values, color="#43a047")
    axes[1].set_title("评分 vs BSR 中位数", fontsize=11)
    axes[1].set_xlabel("评分"); axes[1].tick_params(axis="x", rotation=20)

    # 视频 / A+ 内容（比图片数可靠）
    df["has_video"] = df["video_count"] > 0
    ls1 = df.groupby("has_video")["root_bs_rank"].median()
    ls2 = df.groupby("plus_content")["root_bs_rank"].median()
    g3 = pd.Series({"无视频": ls1.get(False, np.nan), "有视频": ls1.get(True, np.nan),
                    "无A+": ls2.get(False, np.nan), "有A+": ls2.get(True, np.nan)})
    axes[2].bar(g3.index, g3.values, color=["#9e9e9e", "#43a047", "#9e9e9e", "#43a047"])
    axes[2].set_title("视频 / A+ 内容 vs BSR 中位数", fontsize=11)
    axes[2].set_ylabel("BSR 中位数")
    axes[2].tick_params(axis="x", rotation=15)

    plt.suptitle("商品特征与排名（BSR）的关系", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{FIG}/02_bsr_factors.png", bbox_inches="tight")
    plt.close()

    print("\n评论数分档 → BSR 中位数：")
    print(g1.round(0).to_string())
    print("\n评分分档 → BSR 中位数：")
    print(g2.round(0).to_string())
    print("\n图片数分档 → BSR 中位数：")
    print(g3.round(0).to_string())
    return df, corr_s


# ============================================================
# 第 4 步：爆款特征画像
# ============================================================
def hit_product_profile(df):
    print("\n" + "=" * 64)
    print("第 4 步：爆款商品的特征画像")
    print("=" * 64)

    # 定义爆款：BSR Top 20%（排名前 200）
    threshold = df["root_bs_rank"].quantile(0.2)
    df["is_hit"] = df["root_bs_rank"] <= threshold
    print(f"爆款定义：BSR ≤ {threshold:.0f}（前 20%）")
    print(f"爆款数量：{df['is_hit'].sum()} / {len(df)}")

    # 对比爆款 vs 普通
    metrics = ["price", "rating", "reviews_count", "images_count",
               "video_count", "number_of_sellers", "answered_questions"]
    compare = df.groupby("is_hit")[metrics].median().T
    compare.columns = ["普通商品", "爆款"]
    compare["差异"] = ((compare["爆款"] - compare["普通商品"]) /
                       compare["普通商品"].replace(0, np.nan) * 100)
    print("\n爆款 vs 普通商品（中位数对比）：")
    print(compare.round(2).to_string())

    # 图 3：对比柱状图（标准化后对比）
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 左：评论数、图片数、卖家数对比
    show_metrics = ["reviews_count", "images_count", "video_count", "number_of_sellers"]
    x = np.arange(len(show_metrics))
    w = 0.35
    norm = df.groupby("is_hit")[show_metrics].median()
    axes[0].bar(x - w/2, norm.loc[False].values, w, label="普通", color="#9e9e9e")
    axes[0].bar(x + w/2, norm.loc[True].values, w, label="爆款", color="#e53935")
    axes[0].set_xticks(x); axes[0].set_xticklabels(show_metrics, rotation=15)
    axes[0].set_title("爆款 vs 普通：数量类指标（中位数）", fontsize=11)
    axes[0].legend()

    # 右：价格与评分
    show2 = ["price", "rating"]
    x2 = np.arange(len(show2))
    norm2 = df.groupby("is_hit")[show2].median()
    axes[1].bar(x2 - w/2, norm2.loc[False].values, w, label="普通", color="#9e9e9e")
    axes[1].bar(x2 + w/2, norm2.loc[True].values, w, label="爆款", color="#e53935")
    axes[1].set_xticks(x2); axes[1].set_xticklabels(show2)
    axes[1].set_title("爆款 vs 普通：价格与评分", fontsize=11)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{FIG}/03_hit_vs_normal.png", bbox_inches="tight")
    plt.close()

    # 结论
    hits = compare["爆款"]
    print(f"\n结论：爆款商品的评论数是普通商品的 "
          f"{hits['reviews_count']/compare.loc['reviews_count','普通商品']:.1f} 倍，"
          f"评分 {hits['rating']:.2f}（普通 {compare.loc['rating','普通商品']:.2f}）；"
          f"爆款中有视频的占比 {df[df['is_hit']]['video_count'].gt(0).mean():.0%}"
          f"（普通商品 {df[~df['is_hit']]['video_count'].gt(0).mean():.0%}）")
    return df


# ============================================================
# 第 5 步：运营优化建议（价格带 / 折扣 / Listing）
# ============================================================
def optimization_advice(df, bsr_corr_cache):
    print("\n" + "=" * 64)
    print("第 5 步：运营优化建议")
    print("=" * 64)

    # 价格带 vs 表现
    band = df.groupby("price_band", observed=True).agg(
        商品数=("asin", "count"),
        平均BSR=("root_bs_rank", "median"),
        平均评论=("reviews_count", "median"),
        平均评分=("rating", "median"),
    )
    print("价格带表现：")
    print(band.round(1).to_string())

    # 折扣 vs 表现
    disc = df.dropna(subset=["discount_pct"]).copy()
    disc["有折扣"] = disc["discount_pct"] < 0
    d_compare = disc.groupby("有折扣").agg(
        商品数=("asin", "count"),
        BSR中位数=("root_bs_rank", "median"),
        评论中位数=("reviews_count", "median"),
    )
    print("\n有无折扣对比：")
    print(d_compare.round(0).to_string())

    # Listing 质量：图片+视频
    # Listing 完善度：有无视频 + 有无 A+ 内容（图片数字段不可靠，弃用）
    df["listing_quality"] = df["video_count"].clip(0, 1) + df["plus_content"].astype(int)
    ls = df.groupby("listing_quality")["root_bs_rank"].median()
    print("\nListing 完善度（视频+A+）→ BSR 中位数：")
    print(ls.round(0).to_string())

    # 图 4：价格带 vs 平均评论数 & BSR
    fig, ax1 = plt.subplots(figsize=(11, 5))
    x = range(len(band))
    ax1.bar(x, band["商品数"], color="#a8c7e8", label="商品数")
    ax1.set_xticks(list(x)); ax1.set_xticklabels(band.index.astype(str))
    ax1.set_ylabel("商品数", color="#1a5fb4")
    ax1.set_xlabel("价格带")
    ax2 = ax1.twinx()
    ax2.plot(x, band["平均BSR"], color="#e07b39", marker="o", lw=2, label="BSR 中位数")
    ax2.set_ylabel("BSR 中位数（越低越好）", color="#e07b39")
    plt.title("价格带分布与排名表现")
    plt.tight_layout()
    plt.savefig(f"{FIG}/04_price_band.png", bbox_inches="tight")
    plt.close()

    # 图 5：Listing 完善度 vs BSR
    fig, ax = plt.subplots(figsize=(9, 5))
    labels_map = {0: "无视频 + 无A+", 1: "有其一", 2: "视频 + A+ 齐全"}
    xlabels = [labels_map.get(int(i), str(i)) for i in ls.index]
    ax.bar(xlabels, ls.values, color="#43a047")
    ax.set_xlabel("Listing 完善度（视频 + A+ 内容）")
    ax.set_ylabel("BSR 中位数")
    ax.set_title("Listing 完善度与排名表现")
    for i, v in enumerate(ls.values):
        ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{FIG}/05_listing_quality.png", bbox_inches="tight")
    plt.close()

    # 给运营的建议
    print("\n" + "-" * 64)
    print("运营优化建议：")
    print("-" * 64)
    best_band = band["平均BSR"].idxmin()
    print(f"1. 价格带：「{best_band}」的商品 BSR 中位数最低（表现最好）")
    print(f"2. 评论数是最强的排名相关因素（相关系数 {bsr_corr_cache.get('reviews_count', 0):.2f}）")
    print(f"3. Listing 完善度越高，排名越好——建议所有商品配视频 + A+ 内容")
    if len(d_compare) >= 2:
        print(f"4. 有折扣的商品 BSR 中位数 {d_compare.loc[True,'BSR中位数']:.0f}，"
              f"无折扣 {d_compare.loc[False,'BSR中位数']:.0f}")
    return df


# ============================================================
if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)

    df = load_and_clean()
    df = market_overview(df)
    df, bsr_corr_cache = bsr_factors(df)
    df = hit_product_profile(df)
    df = optimization_advice(df, bsr_corr_cache)

    # 保存清洗后的数据
    keep = ["asin", "title", "brand", "main_category", "price", "rating",
            "reviews_count", "root_bs_rank", "images_count", "video_count",
            "number_of_sellers", "discount_pct", "bsr_level", "price_band"]
    df[[c for c in keep if c in df.columns]].to_csv(
        f"{OUT}/amazon_cleaned.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 64)
    print("✅ 分析完成！")
    print(f"   图表：{FIG}/ （5 张）")
    print(f"   数据：{OUT}/amazon_cleaned.csv")
    print("=" * 64)
