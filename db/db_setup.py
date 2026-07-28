# =====================================================================
# Jetson Nano용 분류 결과 DB 초기화 스크립트
# =====================================================================
# 4-1/4-2 추론 스크립트를 실행하면 어차피 자동으로 DB가 만들어지지만,
# 카메라·센서 연결 전에 DB 구조만 먼저 테스트해보고 싶을 때 이 스크립트를
# 따로 실행하면 됩니다. sqlite3는 파이썬 기본 내장이라 별도 설치가 필요
# 없습니다. (DB 서버 설치 X, 그냥 파일 하나로 동작)
#
# 이 스크립트는 db/ 폴더 안에 있고, points.py도 같은 db/ 폴더에 있어야 합니다.
#   project/models/, project/run/, project/db/ (이 스크립트, points.py 위치)
#
# 실행: python3 db_setup.py
# 결과: 이 스크립트와 같은 폴더(db/)에 sorting_log.db 파일 생성
# =====================================================================

import sqlite3
import os

from points import init_users_table

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "sorting_log.db")

# phone: 반납한 사용자 전화번호 (키보드로 입력받음, 식별 안 하면 NULL)
SCHEMA = """
CREATE TABLE IF NOT EXISTS sorting_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,       -- 분류 시각 (ISO 8601 형식)
    phone TEXT,                    -- 반납한 사용자 전화번호 (없으면 NULL, 게스트)
    cnn_class TEXT NOT NULL,       -- CNN이 실제로 예측한 클래스
    cnn_confidence REAL NOT NULL,  -- 예측 확률 (0~1)
    final_class TEXT NOT NULL,     -- 임계값 반영한 최종 클래스 (미확정이면 'unknown')
    points_awarded INTEGER NOT NULL DEFAULT 0  -- 이 건에서 적립된 포인트
)
"""


def main():
    is_new = not os.path.exists(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    conn.execute(SCHEMA)
    init_users_table(conn)  # points.py: users 테이블(전화번호별 누적 포인트) 생성
    conn.commit()

    cursor = conn.execute("SELECT COUNT(*) FROM sorting_log")
    count = cursor.fetchone()[0]
    user_count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    if is_new:
        print(f"새 DB 생성 완료: {os.path.abspath(DB_PATH)}")
    else:
        print(f"기존 DB 확인됨: {os.path.abspath(DB_PATH)} (테이블 이미 있으면 그대로 둠)")

    print(f"현재 저장된 로그 수: {count}개 / 등록된 사용자 수: {user_count}명")

    # 테스트용 더미 데이터 하나 넣어보고 싶으면 아래 주석 해제
    # from datetime import datetime
    # from points import award_points
    # conn.execute(
    #     "INSERT INTO sorting_log (timestamp, phone, cnn_class, cnn_confidence, final_class, points_awarded) "
    #     "VALUES (?, ?, ?, ?, ?, ?)",
    #     (datetime.now().isoformat(), "01012345678", "plastic", 0.87, "plastic", 5),
    # )
    # award_points(conn, "01012345678", "plastic")
    # print("테스트용 더미 데이터 1건 추가함")

    conn.close()


if __name__ == "__main__":
    main()
