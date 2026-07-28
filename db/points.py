# =====================================================================
# 포인트 적립 공용 모듈
# =====================================================================
# db_setup.py, 4-1/4-2 추론 스크립트, db_view.py에서 공통으로 가져다 씁니다.
# 같은 폴더에 이 파일(points.py)을 두면 다른 스크립트에서
# "from points import ..." 로 바로 불러다 쓸 수 있습니다.
# =====================================================================

import sqlite3

DB_PATH = "sorting_log.db"

# 클래스별 지급 포인트 (재활용 난이도/가치에 맞춰 자유롭게 조정하세요)
# 클래스: metal(금속) / plastic(플라스틱) / paper(종이)
POINTS_PER_CLASS = {
    "metal": 10,
    "plastic": 5,
    "paper": 3,
    "unknown": 0,   # 판단 불확실 -> 포인트 없음
}


def init_users_table(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            phone TEXT PRIMARY KEY,
            total_points INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    conn.commit()


def is_valid_phone(text):
    # 아주 단순한 검증: 숫자만, 4자리 이상 (전화번호 뒷자리 등 자유롭게 사용 가능)
    return text.isdigit() and len(text) >= 4


def award_points(conn, phone, cnn_class):
    """분류 결과에 맞는 포인트를 해당 사용자에게 적립. 적립된 포인트 수를 반환."""
    points = POINTS_PER_CLASS.get(cnn_class, 0)

    conn.execute(
        "INSERT INTO users (phone, total_points) VALUES (?, 0) "
        "ON CONFLICT(phone) DO NOTHING",
        (phone,),
    )
    conn.execute(
        "UPDATE users SET total_points = total_points + ? WHERE phone = ?",
        (points, phone),
    )
    conn.commit()
    return points


def get_points(conn, phone):
    cursor = conn.execute("SELECT total_points FROM users WHERE phone = ?", (phone,))
    row = cursor.fetchone()
    return row[0] if row else 0


def get_leaderboard(conn, limit=10):
    cursor = conn.execute(
        "SELECT phone, total_points FROM users ORDER BY total_points DESC LIMIT ?",
        (limit,),
    )
    return cursor.fetchall()
