import os
import sqlite3
from pathlib import Path

from werkzeug.security import check_password_hash, generate_password_hash

try:
    import psycopg2
    import psycopg2.extras
except ImportError:  # pragma: no cover
    psycopg2 = None


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("FRACTUREAI_DATA_DIR", str(BASE_DIR)))
DATA_DIR.mkdir(parents=True, exist_ok=True)
SQLITE_DB_PATH = DATA_DIR / "fractureai.db"
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
USE_POSTGRES = DATABASE_URL.startswith("postgres")


def _normalized_database_url():
    if DATABASE_URL.startswith("postgres://"):
        return DATABASE_URL.replace("postgres://", "postgresql://", 1)
    return DATABASE_URL


def get_connection():
    if USE_POSTGRES:
        if psycopg2 is None:
            return _sqlite_connection()
        try:
            connection = psycopg2.connect(_normalized_database_url())
            connection.autocommit = False
            return connection
        except Exception:
            return _sqlite_connection()

    return _sqlite_connection()


def _sqlite_connection():
    connection = sqlite3.connect(SQLITE_DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def _fetchone_dict(cursor):
    row = cursor.fetchone()
    if row is None:
        return None
    if _connection_uses_postgres(cursor.connection):
        columns = [description[0] for description in cursor.description]
        return dict(zip(columns, row))
    return dict(row)


def _fetchall_dicts(cursor):
    rows = cursor.fetchall()
    if _connection_uses_postgres(cursor.connection):
        columns = [description[0] for description in cursor.description]
        return [dict(zip(columns, row)) for row in rows]
    return [dict(row) for row in rows]


def _connection_uses_postgres(connection):
    return psycopg2 is not None and isinstance(connection, psycopg2.extensions.connection)


def init_db():
    with get_connection() as connection:
        use_postgres = _connection_uses_postgres(connection)
        cursor = connection.cursor()

        if use_postgres:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS cases (
                    id SERIAL PRIMARY KEY,
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
                    confidence DOUBLE PRECISION NOT NULL,
                    severity TEXT,
                    region_label TEXT,
                    health_score DOUBLE PRECISION,
                    overall_health TEXT,
                    uploaded_image TEXT,
                    gradcam_image TEXT,
                    report_file TEXT,
                    feedback TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
        else:
            cursor.executescript(
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

        placeholder = "%s" if use_postgres else "?"
        for username, password, role in (
            ("admin", "admin123", "admin"),
            ("doctor", "doctor123", "doctor"),
        ):
            cursor.execute(
                f"SELECT id FROM users WHERE username = {placeholder}",
                (username,),
            )
            existing = cursor.fetchone()
            if existing is None:
                cursor.execute(
                    f"INSERT INTO users (username, password_hash, role) VALUES ({placeholder}, {placeholder}, {placeholder})",
                    (username, generate_password_hash(password), role),
                )

        if use_postgres:
            cursor.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'cases'
                """
            )
            existing_columns = {row[0] for row in cursor.fetchall()}
        else:
            existing_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(cases)").fetchall()
            }

        column_definitions = {
            "patient_name": "TEXT",
            "patient_id": "TEXT",
            "patient_age": "TEXT",
            "patient_gender": "TEXT",
            "doctor_name": "TEXT",
            "patient_email": "TEXT",
            "doctor_notes": "TEXT",
            "temperature": "TEXT",
            "pulse_rate": "TEXT",
            "spo2": "TEXT",
            "systolic_bp": "TEXT",
            "diastolic_bp": "TEXT",
            "pain_level": "TEXT",
            "symptoms": "TEXT",
            "severity": "TEXT",
            "region_label": "TEXT",
            "health_score": "DOUBLE PRECISION" if use_postgres else "REAL",
            "overall_health": "TEXT",
            "uploaded_image": "TEXT",
            "gradcam_image": "TEXT",
            "report_file": "TEXT",
            "feedback": "TEXT",
        }

        for column_name, column_type in column_definitions.items():
            if column_name not in existing_columns:
                cursor.execute(
                    f"ALTER TABLE cases ADD COLUMN {column_name} {column_type}"
                )

        connection.commit()


def verify_user(username, password):
    with get_connection() as connection:
        placeholder = "%s" if _connection_uses_postgres(connection) else "?"
        cursor = connection.cursor()
        cursor.execute(
            f"SELECT * FROM users WHERE username = {placeholder}",
            (username,),
        )
        user = _fetchone_dict(cursor)

    if user and check_password_hash(user["password_hash"], password):
        return user
    return None


def create_case(case_data):
    columns = list(case_data.keys())
    values = tuple(case_data[column] for column in columns)

    with get_connection() as connection:
        use_postgres = _connection_uses_postgres(connection)
        placeholders = ", ".join([("%s" if use_postgres else "?")] * len(columns))
        cursor = connection.cursor()
        if use_postgres:
            cursor.execute(
                f"INSERT INTO cases ({', '.join(columns)}) VALUES ({placeholders}) RETURNING id",
                values,
            )
            case_id = cursor.fetchone()[0]
        else:
            cursor.execute(
                f"INSERT INTO cases ({', '.join(columns)}) VALUES ({placeholders})",
                values,
            )
            case_id = cursor.lastrowid
        connection.commit()
        return case_id


def list_cases(search_query=""):
    with get_connection() as connection:
        sql = "SELECT * FROM cases"
        params = []
        if search_query:
            wildcard = f"%{search_query}%"
            placeholder = "%s" if _connection_uses_postgres(connection) else "?"
            sql += f" WHERE patient_name LIKE {placeholder} OR patient_id LIKE {placeholder}"
            params.extend([wildcard, wildcard])
        sql += " ORDER BY id DESC"
        cursor = connection.cursor()
        cursor.execute(sql, params)
        return _fetchall_dicts(cursor)


def recent_cases(limit=5):
    with get_connection() as connection:
        placeholder = "%s" if _connection_uses_postgres(connection) else "?"
        cursor = connection.cursor()
        cursor.execute(
            f"SELECT * FROM cases ORDER BY id DESC LIMIT {placeholder}",
            (limit,),
        )
        return _fetchall_dicts(cursor)


def get_case(case_id):
    with get_connection() as connection:
        placeholder = "%s" if _connection_uses_postgres(connection) else "?"
        cursor = connection.cursor()
        cursor.execute(
            f"SELECT * FROM cases WHERE id = {placeholder}",
            (case_id,),
        )
        return _fetchone_dict(cursor)


def update_feedback(case_id, feedback):
    with get_connection() as connection:
        placeholder = "%s" if _connection_uses_postgres(connection) else "?"
        cursor = connection.cursor()
        cursor.execute(
            f"UPDATE cases SET feedback = {placeholder} WHERE id = {placeholder}",
            (feedback, case_id),
        )
        connection.commit()


def update_report_file(case_id, report_file):
    with get_connection() as connection:
        placeholder = "%s" if _connection_uses_postgres(connection) else "?"
        cursor = connection.cursor()
        cursor.execute(
            f"UPDATE cases SET report_file = {placeholder} WHERE id = {placeholder}",
            (report_file, case_id),
        )
        connection.commit()


def dashboard_stats():
    with get_connection() as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM cases")
        total_cases = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM cases WHERE result = 'Fracture'")
        fracture_cases = cursor.fetchone()[0]
        cursor.execute("SELECT AVG(confidence) FROM cases")
        average_confidence = cursor.fetchone()[0]

    safe_total = total_cases or 1
    return {
        "total_cases": total_cases,
        "fracture_cases": fracture_cases,
        "normal_cases": total_cases - fracture_cases,
        "fracture_rate": round((fracture_cases / safe_total) * 100, 2) if total_cases else 0,
        "average_confidence": round(float(average_confidence or 0), 2),
    }
