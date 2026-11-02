import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from preprocessing import FEATURE_COLUMNS, TARGET

PICKUP_HOUR = "pickup_hour"
DISTANCE = "straight_line_km_approx"


def get_metric(results, *keys):
    for key in keys:
        if key in results:
            return float(results[key])
    raise ValueError(f"Validation results are missing metric: {' or '.join(keys)}")


def generate_report(train_path, results_path, output_path):
    data_columns = [*FEATURE_COLUMNS, "trip_id", TARGET]
    data = pd.read_csv(train_path, usecols=data_columns)
    if data.empty:
        raise ValueError("Training data is empty.")
    if data["trip_id"].isna().any() or data["trip_id"].duplicated().any():
        raise ValueError("Training trip_id values must be present and unique.")
    if data[TARGET].isna().any() or not np.isfinite(data[TARGET].to_numpy(dtype=float)).all():
        raise ValueError("Training target contains missing or non-finite values.")
    if (data[TARGET] <= 0).any():
        raise ValueError("Training target must contain only positive durations.")

    with Path(results_path).open(encoding="utf-8") as file:
        results = json.load(file)
    baseline_score = get_metric(
        results, "baseline_median_rmsle", "validation_rmsle_median_baseline"
    )
    model_score = get_metric(
        results, "lightgbm_rmsle", "validation_rmsle_lightgbm"
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration = data[TARGET]
    missing_cells = int(data.isna().sum().sum())
    quantiles = duration.quantile([0.5, 0.9, 0.99])
    hour_medians = data.groupby(PICKUP_HOUR, observed=True)[TARGET].median()
    distance_bins = pd.cut(data[DISTANCE], bins=10, duplicates="drop")
    distance_medians = data.groupby(distance_bins, observed=True)[TARGET].median()
    lowest_hour = int(hour_medians.idxmin())
    highest_hour = int(hour_medians.idxmax())
    first_distance_median = float(distance_medians.iloc[0])
    last_distance_median = float(distance_medians.iloc[-1])

    with PdfPages(output_path) as pdf:
        figure, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
        figure.suptitle("UNETI14 — Khám phá dữ liệu taxi NYC", fontsize=18, fontweight="bold")

        bins = np.logspace(np.log10(duration.min()), np.log10(duration.max()), 45)
        axes[0, 0].hist(duration, bins=bins, color="#4472C4", edgecolor="white")
        axes[0, 0].set_xscale("log")
        axes[0, 0].set_yscale("log")
        axes[0, 0].set_title("1. Phân phối thời gian chuyến")
        axes[0, 0].set_xlabel("Thời gian (giây, trục log)")
        axes[0, 0].set_ylabel("Số chuyến (trục log)")
        axes[0, 0].grid(alpha=0.2)

        axes[0, 1].plot(hour_medians.index, hour_medians.values, marker="o", color="#ED7D31")
        axes[0, 1].set_title("2. Trung vị thời gian theo giờ đón")
        axes[0, 1].set_xlabel("Giờ đón")
        axes[0, 1].set_ylabel("Trung vị (giây)")
        axes[0, 1].set_xticks(range(0, 24, 2))
        axes[0, 1].grid(alpha=0.2)

        interval_midpoints = [interval.mid for interval in distance_medians.index]
        axes[1, 0].plot(
            interval_midpoints,
            distance_medians.values,
            marker="o",
            color="#70AD47",
        )
        axes[1, 0].set_title("3. Trung vị thời gian theo khoảng cách")
        axes[1, 0].set_xlabel("Khoảng cách đường chim bay (km)")
        axes[1, 0].set_ylabel("Trung vị (giây)")
        axes[1, 0].grid(alpha=0.2)

        axes[1, 1].axis("off")
        axes[1, 1].text(
            0,
            0.95,
            (
                f"Số bản ghi: {len(data):,}\n"
                f"Số cột trong Train: {len(data_columns)}\n"
                f"Giá trị thiếu: {missing_cells:,}\n"
                f"Trip ID trùng: {data['trip_id'].duplicated().sum():,}\n\n"
                f"Trung vị: {quantiles.loc[0.5]:,.0f} giây\n"
                f"P90: {quantiles.loc[0.9]:,.0f} giây\n"
                f"P99: {quantiles.loc[0.99]:,.0f} giây\n\n"
                "Thời gian lệch phải; dùng log1p cho nhãn.\n"
                "Không đưa trip_id vào đặc trưng."
            ),
            va="top",
            fontsize=12,
            linespacing=1.5,
        )
        figure.tight_layout(rect=(0.03, 0.03, 0.97, 0.93))
        pdf.savefig(figure)
        plt.close(figure)

        figure = plt.figure(figsize=(8.27, 11.69))
        figure.suptitle("Mô hình, đánh giá và kết luận", fontsize=18, fontweight="bold", y=0.97)
        figure.text(
            0.08,
            0.91,
            (
                f"Tập train: {len(data):,} chuyến; thiếu {missing_cells:,} ô; "
                f"trip_id duy nhất. Nhãn trip_duration dương, đơn vị giây.\n"
                "Đặc trưng gồm thông tin lúc đón, khoảng cách và tọa độ đã làm tròn; "
                "loại trip_id/nhãn khỏi đầu vào.\n"
                "Tạo delta kinh/vĩ độ và mã hóa chu kỳ giờ/ngày bằng sin-cos; "
                "biến rời rạc được khai báo dạng category.\n"
                "Đánh giá trên validation 15% tách ngẫu nhiên (random_state=42), "
                "tối ưu RMSLE trên log1p(trip_duration)."
            ),
            va="top",
            fontsize=11,
            linespacing=1.55,
            wrap=True,
        )

        table = figure.add_axes((0.08, 0.65, 0.84, 0.12))
        table.axis("off")
        table.table(
            cellText=[
                ["Baseline trung vị", f"{baseline_score:.6f}", "Dự đoán trung vị train"],
                ["LightGBM", f"{model_score:.6f}", "Mô hình được chọn"],
            ],
            colLabels=["Phương pháp", "Validation RMSLE", "Ghi chú"],
            cellLoc="center",
            loc="center",
            colWidths=[0.28, 0.26, 0.46],
        )

        figure.text(
            0.08,
            0.59,
            (
                "Nhận xét từ dữ liệu:\n"
                f"• Thời gian chuyến lệch phải: trung vị {quantiles.loc[0.5]:,.0f}s, "
                f"P99 {quantiles.loc[0.99]:,.0f}s.\n"
                f"• Trung vị theo giờ thấp nhất ở {lowest_hour}h và cao nhất ở {highest_hour}h.\n"
                f"• Trung vị ở nhóm khoảng cách nhỏ nhất/lớn nhất lần lượt "
                f"{first_distance_median:,.0f}s và {last_distance_median:,.0f}s."
            ),
            va="top",
            fontsize=11,
            linespacing=1.5,
        )
        selection = (
            "Kết luận: "
            "chọn LightGBM vì validation RMSLE thấp hơn baseline trung vị."
            if model_score < baseline_score
            else "Kết luận: baseline trung vị có RMSLE thấp hơn; cần xem lại lựa chọn model."
        )
        figure.text(
            0.08,
            0.39,
            (
                f"{selection}\n\n"
                "Hạn chế: tọa độ/khoảng cách đã được làm tròn; validation ngẫu nhiên "
                "không bảo đảm phản ánh hoàn toàn bộ test riêng. Không loại bỏ nhãn "
                "cực trị nếu chưa có căn cứ dữ liệu.\n\n"
                "Phân công thành viên: bổ sung họ tên và nhiệm vụ của 03 thành viên trước khi nộp."
            ),
            va="top",
            fontsize=11,
            linespacing=1.6,
            wrap=True,
        )
        figure.text(0.08, 0.07, "UNETI Olympic AI 2026 · Vòng 2 — AI & Data Challenge", fontsize=9)
        pdf.savefig(figure)
        plt.close(figure)

    print(f"Saved report: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate the UNETI14 EDA and model report PDF.")
    parser.add_argument("--train", default="data/UNETI2026_TRAIN_CHUNG_1200000.csv")
    parser.add_argument("--results", default="outputs/validation_results.json")
    parser.add_argument("--output", default="outputs/UNETI14_report.pdf")
    args = parser.parse_args()
    generate_report(args.train, args.results, args.output)


if __name__ == "__main__":
    main()
