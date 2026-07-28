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
#
# 이 스크립트는 db/ 폴더 안에 있고, points.py도 같은 db/ 폴더에 있어야 합니다.
# =====================================================================

import os
import sqlite3
import sys
import csv

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
    else:  # recent (기본값): 최근 20개만
        print_rows(rows[-20:])
        print_stats(rows)
        print_leaderboard(conn)

    conn.close()


if __name__ == "__main__":
    main()
