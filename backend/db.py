import sqlite3
import os
from pathlib import Path

from werkzeug.security import check_password_hash, generate_password_hash


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("FRACTUREAI_DATA_DIR", str(BASE_DIR)))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "fractureai.db"


def get_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    with get_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_name TEXT,
                patient_id TEXT,
                patient_age TEXT,
                patient_gender TEXT,
                doctor_name TEXT,
                patient_email TEXT,
                doctor_notes TEXT,
                temperature TEXT,
                pulse_rate TEXT,
                spo2 TEXT,
                systolic_bp TEXT,
                diastolic_bp TEXT,
                pain_level TEXT,
                symptoms TEXT,
                result TEXT NOT NULL,
                confidence REAL NOT NULL,
                severity TEXT,
                region_label TEXT,
                health_score REAL,
                overall_health TEXT,
                uploaded_image TEXT,
                gradcam_image TEXT,
                report_file TEXT,
                feedback TEXT,
                created_at TEXT NOT NULL
            );
            """
        )

        for username, password, role in (
            ("admin", "admin123", "admin"),
            ("doctor", "doctor123", "doctor"),
        ):
            existing = connection.execute(
                "SELECT id FROM users WHERE username = ?",
                (username,),
            ).fetchone()
            if existing is None:
                connection.execute(
                    "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                    (username, generate_password_hash(password), role),
                )

        existing_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(cases)").fetchall()
        }
        column_definitions = {
            "temperature": "TEXT",
            "pulse_rate": "TEXT",
            "spo2": "TEXT",
            "systolic_bp": "TEXT",
            "diastolic_bp": "TEXT",
            "pain_level": "TEXT",
            "symptoms": "TEXT",
            "health_score": "REAL",
            "overall_health": "TEXT",
        }
        for column_name, column_type in column_definitions.items():
            if column_name not in existing_columns:
                connection.execute(
                    f"ALTER TABLE cases ADD COLUMN {column_name} {column_type}"
                )

        connection.commit()


def verify_user(username, password):
    with get_connection() as connection:
        user = connection.execute(
            "SELECT * FROM users WHERE username = ?",
            (username,),
        ).fetchone()

    if user and check_password_hash(user["password_hash"], password):
        return dict(user)
    return None


def create_case(case_data):
    columns = ", ".join(case_data.keys())
    placeholders = ", ".join(["?"] * len(case_data))
    values = tuple(case_data.values())

    with get_connection() as connection:
        cursor = connection.execute(
            f"INSERT INTO cases ({columns}) VALUES ({placeholders})",
            values,
        )
        connection.commit()
        return cursor.lastrowid


def list_cases(search_query=""):
    sql = "SELECT * FROM cases"
    params = []

    if search_query:
        sql += " WHERE patient_name LIKE ? OR patient_id LIKE ?"
        wildcard = f"%{search_query}%"
        params.extend([wildcard, wildcard])

    sql += " ORDER BY datetime(created_at) DESC"

    with get_connection() as connection:
        rows = connection.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def recent_cases(limit=5):
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT * FROM cases ORDER BY datetime(created_at) DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_case(case_id):
    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM cases WHERE id = ?",
            (case_id,),
        ).fetchone()
    return dict(row) if row else None


def update_feedback(case_id, feedback):
    with get_connection() as connection:
        connection.execute(
            "UPDATE cases SET feedback = ? WHERE id = ?",
            (feedback, case_id),
        )
        connection.commit()


def update_report_file(case_id, report_file):
    with get_connection() as connection:
        connection.execute(
            "UPDATE cases SET report_file = ? WHERE id = ?",
            (report_file, case_id),
        )
        connection.commit()


def dashboard_stats():
    with get_connection() as connection:
        total_cases = connection.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
        fracture_cases = connection.execute(
            "SELECT COUNT(*) FROM cases WHERE result = 'Fracture'"
        ).fetchone()[0]
        average_confidence = connection.execute(
            "SELECT AVG(confidence) FROM cases"
        ).fetchone()[0]

    safe_total = total_cases or 1
    return {
        "total_cases": total_cases,
        "fracture_cases": fracture_cases,
        "normal_cases": total_cases - fracture_cases,
        "fracture_rate": round((fracture_cases / safe_total) * 100, 2) if total_cases else 0,
        "average_confidence": round(average_confidence or 0, 2),
    }
