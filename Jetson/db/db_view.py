# =====================================================================
# Jetson Nano 분류 결과 DB 조회 스크립트
# =====================================================================
# sorting_log.db에 쌓인 기록을 확인하거나 CSV로 내보낼 때 사용합니다.
# 실환경 테스트할 때 "실제로 몇 개 중 몇 개를 맞췄는지" 확인하는 용도로도
# 쓰고(계획서 평가지표 - 실환경 정분류율), 포인트 적립 현황도 확인합니다.
#
# 사용법
#   python3 db_view.py                     최근 20건 + 클래스별 통계 + 포인트 랭킹
#   python3 db_view.py all                 전체 기록 다 출력
#   python3 db_view.py export              sorting_log.csv 로 내보내기
#   python3 db_view.py points              사용자별 포인트 랭킹만 출력
#   python3 db_view.py points test_sorting_log.db   다른 DB 파일 지정 (테스트용 DB 확인 등)
#   python3 db_view.py timing              추론 시간(infer_ms) 평균/중앙값/최소/최대 통계
#                                           (4-1~4-5 스크립트가 기록한 infer_ms 기준.
#                                            4-4는 yolo_ms/cnn_ms 세부 분해도 함께 출력)
#   python3 db_view.py method              하이브리드(gui_dashboard_hybrid.py) 판정 경로 분포
#                                           - 몇 %가 YOLO 1회로 끝나고 몇 %가 CNN까지 갔는지
#   python3 db_view.py throughput          연속 투입 테스트 구간의 처리량(개/분, 개/시간)
#                                           - timestamp 간격이 30초 이하인 구간만 집계
#
# 이 스크립트는 db/ 폴더 안에 있고, points.py도 같은 db/ 폴더에 있어야 합니다.
# =====================================================================

import os
import sqlite3
import statistics
import sys
import csv
from datetime import datetime

from points import get_leaderboard

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "sorting_log.db")


def fetch_all(conn):
    cursor = conn.execute(
        "SELECT id, timestamp, phone, cnn_class, cnn_confidence, final_class, points_awarded "
        "FROM sorting_log ORDER BY id"
    )
    return cursor.fetchall()


def print_rows(rows):
    print(f"{'ID':<5}{'시각':<21}{'전화번호':<13}{'CNN 예측':<10}{'확신도':<8}{'최종':<10}{'포인트':<6}")
    print("-" * 76)
    for row_id, ts, phone, cnn_class, conf, final_class, points in rows:
        phone_disp = phone if phone else "-"
        ts_disp = ts[:19]  # 마이크로초 잘라서 보기 좋게
        print(f"{row_id:<5}{ts_disp:<21}{phone_disp:<13}{cnn_class:<10}{conf:<8.2f}{final_class:<10}{points:<6}")


def print_stats(rows):
    if not rows:
        print("\n기록이 없습니다.")
        return

    counts = {}
    for _, _, _, _, _, final_class, _ in rows:
        counts[final_class] = counts.get(final_class, 0) + 1

    total = len(rows)
    print(f"\n총 {total}건")
    print("-" * 30)
    for cls, n in sorted(counts.items(), key=lambda x: -x[1]):
        pct = n / total * 100
        print(f"{cls:<12} {n:>4}건  ({pct:5.1f}%)")


def print_leaderboard(conn):
    board = get_leaderboard(conn, limit=10)
    if not board:
        print("\n등록된 사용자가 없습니다.")
        return

    print("\n포인트 랭킹 (상위 10명)")
    print("-" * 30)
    for rank, (phone, points) in enumerate(board, start=1):
        print(f"{rank:>2}. {phone:<15} {points:>6}점")


def print_timing(conn):
    cols = [row[1] for row in conn.execute("PRAGMA table_info(sorting_log)")]
    if "infer_ms" not in cols:
        print(
            "\n이 DB에는 infer_ms 컬럼이 없습니다. "
            "4-1~4-5 스크립트를 최신 버전으로 다시 실행해서 데이터를 쌓아주세요."
        )
        return

    values = [
        row[0]
        for row in conn.execute("SELECT infer_ms FROM sorting_log WHERE infer_ms IS NOT NULL")
    ]
    if not values:
        print("\n추론 시간 기록이 없습니다 (infer_ms가 전부 비어 있음). 먼저 실기기에서 몇 건 돌려보세요.")
        return

    print(f"\n추론 시간(infer_ms) 통계 — {len(values)}건")
    print("-" * 30)
    print(f"평균     {statistics.mean(values):7.1f} ms")
    print(f"중앙값   {statistics.median(values):7.1f} ms")
    print(f"최소     {min(values):7.1f} ms")
    print(f"최대     {max(values):7.1f} ms")

    # 4-4(하이브리드)는 yolo_ms/cnn_ms를 따로 기록하므로, 있으면 세부 분해도 보여줌
    if "yolo_ms" in cols and "cnn_ms" in cols:
        yc_rows = conn.execute(
            "SELECT yolo_ms, cnn_ms FROM sorting_log WHERE yolo_ms IS NOT NULL AND cnn_ms IS NOT NULL"
        ).fetchall()
        if yc_rows:
            yolo_vals = [r[0] for r in yc_rows]
            cnn_vals = [r[1] for r in yc_rows]
            print(f"\n하이브리드(4-4) 단계별 분해 — {len(yc_rows)}건")
            print("-" * 30)
            print(f"YOLO 평균  {statistics.mean(yolo_vals):7.1f} ms")
            print(f"CNN  평균  {statistics.mean(cnn_vals):7.1f} ms")


def print_method_breakdown(conn):
    """gui_dashboard_hybrid.py가 기록하는 method 컬럼("YOLO" / "CNN(모델명)" /
    "YOLO+CNN 둘 다 확신도 부족") 분포를 보여줌. 하이브리드 설계의 실제 효과 —
    "평시엔 YOLO 1회로 빠르게, 애매할 때만 CNN까지 이중 검증" — 를 숫자로 확인하는 용도."""
    cols = [row[1] for row in conn.execute("PRAGMA table_info(sorting_log)")]
    if "method" not in cols:
        print(
            "\n이 DB에는 method 컬럼이 없습니다. "
            "gui_dashboard_hybrid.py를 최신 버전으로 다시 실행해서 데이터를 쌓아주세요."
        )
        return

    rows = conn.execute(
        "SELECT method FROM sorting_log WHERE method IS NOT NULL AND method != ''"
    ).fetchall()
    if not rows:
        print("\nmethod 기록이 없습니다. 먼저 gui_dashboard_hybrid.py로 몇 건 트리거해보세요.")
        return

    counts = {}
    for (m,) in rows:
        counts[m] = counts.get(m, 0) + 1
    total = len(rows)

    print(f"\n하이브리드 판정 경로 분포 — {total}건")
    print("-" * 45)
    for m, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"{m:<30} {n:>4}건 ({n / total * 100:5.1f}%)")

    yolo_only = counts.get("YOLO", 0)
    print(f"\nYOLO 1회로 종료(CNN 미실행): {yolo_only}/{total}건 ({yolo_only / total * 100:.1f}%)")
    print("-> 이 비율이 높을수록 하이브리드 구조가 평시에 CNN 없이 더 빠르게 끝난다는 뜻")


def print_throughput(conn, gap_threshold_sec=30):
    """timestamp 간격을 기준으로 처리량(개/분)을 계산. 세션 사이 긴 공백(사람이
    자리를 비운 시간 등)은 gap_threshold_sec 초과 구간으로 판단해 제외하고,
    "연속으로 물체를 투입한 구간"만 모아서 평균 처리 간격을 계산합니다."""
    ts_rows = conn.execute("SELECT timestamp FROM sorting_log ORDER BY id").fetchall()
    if len(ts_rows) < 2:
        print("\n처리량을 계산하려면 최소 2건 이상의 기록이 필요합니다.")
        return

    timestamps = [datetime.fromisoformat(row[0]) for row in ts_rows]
    deltas = [
        (timestamps[i + 1] - timestamps[i]).total_seconds()
        for i in range(len(timestamps) - 1)
    ]
    continuous = [d for d in deltas if 0 < d <= gap_threshold_sec]

    if not continuous:
        print(
            f"\n간격이 {gap_threshold_sec}초 이하인 연속 구간이 없습니다. "
            "물체를 연속으로 투입하는 테스트를 한 뒤 다시 확인해보세요."
        )
        return

    mean_delta = statistics.mean(continuous)
    print(f"\n처리량 — 연속 구간 {len(continuous)}건 기준 (간격 {gap_threshold_sec}초 초과 구간은 제외)")
    print("-" * 45)
    print(f"평균 처리 간격   {mean_delta:6.1f} 초/개")
    print(f"처리량           {60 / mean_delta:6.1f} 개/분   ({3600 / mean_delta:.0f} 개/시간)")
    print(f"최소 간격        {min(continuous):6.1f} 초/개")
    print(f"최대 간격        {max(continuous):6.1f} 초/개")


def export_csv(rows):
    out_path = os.path.join(BASE_DIR, "sorting_log.csv")
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["id", "timestamp", "phone", "cnn_class", "cnn_confidence", "final_class", "points_awarded"]
        )
        writer.writerows(rows)
    print(f"CSV로 내보냄: {out_path} ({len(rows)}건)")


def main():
    # 두 번째 인자로 DB 경로를 직접 지정할 수 있음 (파일명만 주면 db/ 폴더 기준으로 찾음)
    # 예: python3 db_view.py points test_sorting_log.db
    if len(sys.argv) > 2:
        db_path = sys.argv[2]
        if not os.path.isabs(db_path):
            db_path = os.path.join(BASE_DIR, db_path)
    else:
        db_path = DB_PATH
    conn = sqlite3.connect(db_path)
    rows = fetch_all(conn)

    mode = sys.argv[1] if len(sys.argv) > 1 else "recent"

    if mode == "export":
        export_csv(rows)
    elif mode == "all":
        print_rows(rows)
        print_stats(rows)
        print_leaderboard(conn)
    elif mode == "points":
        print_leaderboard(conn)
    elif mode == "timing":
        print_timing(conn)
    elif mode == "method":
        print_method_breakdown(conn)
    elif mode == "throughput":
        print_throughput(conn)
    else:  # recent (기본값): 최근 20개만
        print_rows(rows[-20:])
        print_stats(rows)
        print_leaderboard(conn)

    conn.close()


if __name__ == "__main__":
    main()
