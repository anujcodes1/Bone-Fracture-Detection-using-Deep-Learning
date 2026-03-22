from pathlib import Path
from datetime import datetime
from email.message import EmailMessage
from functools import wraps
import os
import smtplib
import sys
from uuid import uuid4

import numpy as np
from flask import Flask, redirect, render_template, request, send_from_directory, session, url_for
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "model"
DATASET_DIR = BASE_DIR / "dataset"
MODEL_PATH = MODEL_DIR / "bone_fracture_model.h5"
TRAINING_GRAPH_PATH = MODEL_DIR / "training_history.png"
DATA_DIR = Path(os.getenv("FRACTUREAI_DATA_DIR", str(BASE_DIR)))
UPLOAD_DIR = DATA_DIR / "uploads"
GRADCAM_DIR = DATA_DIR / "outputs" / "gradcam"
PDF_DIR = DATA_DIR / "reports" / "pdf"
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}

sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from db import create_case, dashboard_stats, get_case, init_db, list_cases, recent_cases, update_feedback, verify_user  # noqa: E402
from generate_gradcam import generate_gradcam  # noqa: E402
from predict_single_image import predict_fracture  # noqa: E402
from preprocess_data import create_data_generators  # noqa: E402


app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "backend" / "templates"),
    static_folder=str(BASE_DIR / "static"),
)
app.secret_key = os.getenv("FRACTUREAI_SECRET_KEY", "fractureai-secret-key")
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)

for directory in (DATA_DIR, UPLOAD_DIR, GRADCAM_DIR, PDF_DIR):
    directory.mkdir(parents=True, exist_ok=True)

init_db()


def login_required(view_function):
    @wraps(view_function)
    def wrapped_view(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return view_function(*args, **kwargs)

    return wrapped_view


def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def dataset_is_ready():
    required_dirs = [
        DATASET_DIR / "train" / "fractured",
        DATASET_DIR / "train" / "non_fractured",
        DATASET_DIR / "val" / "fractured",
        DATASET_DIR / "val" / "non_fractured",
        DATASET_DIR / "test" / "fractured",
        DATASET_DIR / "test" / "non_fractured",
    ]
    return all(path.exists() for path in required_dirs)


def build_patient_details(form_data):
    return {
        "patient_name": form_data.get("patient_name", "").strip(),
        "patient_id": form_data.get("patient_id", "").strip(),
        "patient_age": form_data.get("patient_age", "").strip(),
        "patient_gender": form_data.get("patient_gender", "").strip(),
        "doctor_name": form_data.get("doctor_name", "").strip(),
        "patient_email": form_data.get("patient_email", "").strip(),
        "doctor_notes": form_data.get("doctor_notes", "").strip(),
        "temperature": form_data.get("temperature", "").strip(),
        "pulse_rate": form_data.get("pulse_rate", "").strip(),
        "spo2": form_data.get("spo2", "").strip(),
        "systolic_bp": form_data.get("systolic_bp", "").strip(),
        "diastolic_bp": form_data.get("diastolic_bp", "").strip(),
        "pain_level": form_data.get("pain_level", "").strip(),
        "symptoms": form_data.get("symptoms", "").strip(),
    }


def patient_details_defaults(details=None):
    details = details or {}
    return {
        "patient_name": details.get("patient_name", ""),
        "patient_id": details.get("patient_id", ""),
        "patient_age": details.get("patient_age", ""),
        "patient_gender": details.get("patient_gender", ""),
        "doctor_name": details.get("doctor_name", ""),
        "patient_email": details.get("patient_email", ""),
        "doctor_notes": details.get("doctor_notes", ""),
        "temperature": details.get("temperature", ""),
        "pulse_rate": details.get("pulse_rate", ""),
        "spo2": details.get("spo2", ""),
        "systolic_bp": details.get("systolic_bp", ""),
        "diastolic_bp": details.get("diastolic_bp", ""),
        "pain_level": details.get("pain_level", ""),
        "symptoms": details.get("symptoms", ""),
    }


def estimate_severity(result, confidence):
    if result != "Fracture":
        return "Low Risk"
    if confidence >= 90:
        return "Severe Suspicion"
    if confidence >= 75:
        return "Moderate Suspicion"
    return "Mild Suspicion"


def estimate_region_label(filename):
    lowered = filename.lower()
    region_keywords = {
        "wrist": "Possible Wrist Region",
        "hand": "Possible Hand Region",
        "leg": "Possible Leg Region",
        "shoulder": "Possible Shoulder Region",
        "hip": "Possible Hip Region",
        "ankle": "Possible Ankle Region",
    }
    for keyword, label in region_keywords.items():
        if keyword in lowered:
            return label
    return "General Bone Region"


def compute_health_assessment(patient_details, result, confidence):
    score = 100
    flags = []

    def parse_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    temperature = parse_float(patient_details.get("temperature"))
    pulse_rate = parse_float(patient_details.get("pulse_rate"))
    spo2 = parse_float(patient_details.get("spo2"))
    systolic = parse_float(patient_details.get("systolic_bp"))
    diastolic = parse_float(patient_details.get("diastolic_bp"))
    pain_level = parse_float(patient_details.get("pain_level"))

    if temperature is not None and (temperature < 36.1 or temperature > 37.5):
        score -= 8
        flags.append("Temperature out of normal range")

    if pulse_rate is not None and (pulse_rate < 60 or pulse_rate > 100):
        score -= 10
        flags.append("Pulse rate needs attention")

    if spo2 is not None and spo2 < 95:
        score -= 18
        flags.append("Oxygen saturation is low")

    if systolic is not None and diastolic is not None:
        if systolic > 140 or diastolic > 90 or systolic < 90 or diastolic < 60:
            score -= 14
            flags.append("Blood pressure is outside the healthy range")

    if pain_level is not None:
        score -= min(max(pain_level, 0), 10) * 2
        if pain_level >= 7:
            flags.append("High pain level reported")

    if result == "Fracture":
        score -= 18
        flags.append("Fracture detected by AI screening")

    if confidence >= 90 and result == "Fracture":
        score -= 6

    score = max(0, round(score, 2))

    if score >= 85:
        overall_health = "Stable"
    elif score >= 65:
        overall_health = "Needs Attention"
    else:
        overall_health = "Critical Review Recommended"

    return score, overall_health, flags


def create_pdf_report(case_data):
    report_name = f"fracture_report_{uuid4().hex}.pdf"
    report_path = PDF_DIR / report_name

    pdf = canvas.Canvas(str(report_path), pagesize=A4)
    width, height = A4

    pdf.setFillColor(colors.HexColor("#08131f"))
    pdf.rect(0, 0, width, height, fill=1, stroke=0)

    pdf.setFillColor(colors.HexColor("#82f6ff"))
    pdf.setFont("Helvetica-Bold", 24)
    pdf.drawString(36, height - 46, "FractureAI Full Case Summary")

    pdf.setFillColor(colors.HexColor("#d7f8ff"))
    pdf.setFont("Helvetica", 11)
    pdf.drawString(36, height - 68, f"Generated: {case_data['created_at']}")
    pdf.drawString(36, height - 84, "Academic AI report for bone fracture screening.")

    pdf.setStrokeColor(colors.HexColor("#2de0c2"))
    pdf.line(36, height - 96, width - 36, height - 96)

    sections = [
        ("Patient Name", case_data["patient_name"] or "Not Provided"),
        ("Patient ID", case_data["patient_id"] or "Not Provided"),
        ("Age", case_data["patient_age"] or "Not Provided"),
        ("Gender", case_data["patient_gender"] or "Not Provided"),
        ("Doctor", case_data["doctor_name"] or "Not Provided"),
        ("Patient Email", case_data["patient_email"] or "Not Provided"),
        ("Temperature", case_data["temperature"] or "Not Provided"),
        ("Pulse Rate", case_data["pulse_rate"] or "Not Provided"),
        ("SpO2", case_data["spo2"] or "Not Provided"),
        ("Blood Pressure", f"{case_data['systolic_bp'] or 'NA'} / {case_data['diastolic_bp'] or 'NA'}"),
        ("Pain Level", case_data["pain_level"] or "Not Provided"),
        ("Result", case_data["result"]),
        ("Confidence", f"{case_data['confidence']:.2f}%"),
        ("Severity", case_data["severity"]),
        ("Region Label", case_data["region_label"]),
        ("Health Score", f"{case_data['health_score']:.2f}/100"),
        ("Overall Health", case_data["overall_health"]),
        ("Symptoms", case_data["symptoms"] or "Not Provided"),
        ("Doctor Notes", case_data["doctor_notes"] or "No notes added"),
    ]

    y_pos = height - 130
    for label, value in sections:
        pdf.setFillColor(colors.HexColor("#96b5c7"))
        pdf.setFont("Helvetica-Bold", 11)
        pdf.drawString(36, y_pos, f"{label}:")
        pdf.setFillColor(colors.white)
        pdf.setFont("Helvetica", 11)
        pdf.drawString(150, y_pos, str(value)[:80])
        y_pos -= 18

    image_y = 180
    image_width = 240
    image_height = 170

    def draw_report_image(image_path, x_pos, title):
        if not image_path or not image_path.exists():
            return
        pdf.setFillColor(colors.white)
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(x_pos, image_y + image_height + 16, title)
        pdf.drawImage(
            ImageReader(str(image_path)),
            x_pos,
            image_y,
            width=image_width,
            height=image_height,
            preserveAspectRatio=True,
            mask="auto",
        )

    draw_report_image(UPLOAD_DIR / case_data["uploaded_image"], 36, "Uploaded X-ray")
    if case_data["gradcam_image"]:
        draw_report_image(GRADCAM_DIR / case_data["gradcam_image"], 310, "Grad-CAM Heatmap")

    pdf.setFillColor(colors.HexColor("#96b5c7"))
    pdf.setFont("Helvetica", 10)
    pdf.drawString(36, 38, "Clinical decisions should always be reviewed by a qualified medical professional.")
    pdf.save()
    return report_name


def send_report_email(case_record, recipient=None):
    recipient = (recipient or case_record.get("patient_email") or "").strip()
    if not recipient:
        return False, "Please enter an email address to send the report."

    smtp_server = os.getenv("FRACTUREAI_SMTP_SERVER")
    smtp_port = os.getenv("FRACTUREAI_SMTP_PORT", "587")
    smtp_username = os.getenv("FRACTUREAI_SMTP_USERNAME")
    smtp_password = os.getenv("FRACTUREAI_SMTP_PASSWORD")
    sender_email = os.getenv("FRACTUREAI_SENDER_EMAIL", smtp_username or "")

    if not all([smtp_server, smtp_username, smtp_password, sender_email]):
        return False, "Email server is not configured. Add FRACTUREAI_SMTP_* environment variables."

    message = EmailMessage()
    message["Subject"] = f"FractureAI Case Report for {case_record['patient_name'] or 'Patient'}"
    message["From"] = sender_email
    message["To"] = recipient
    message.set_content(
        f"Result: {case_record['result']}\n"
        f"Confidence: {case_record['confidence']:.2f}%\n"
        f"Severity: {case_record['severity']}\n"
        f"Health Score: {case_record['health_score']:.2f}/100\n"
        f"Overall Health: {case_record['overall_health']}\n"
        f"Generated: {case_record['created_at']}"
    )

    report_path = PDF_DIR / case_record["report_file"]
    if report_path.exists():
        with report_path.open("rb") as report_obj:
            message.add_attachment(
                report_obj.read(),
                maintype="application",
                subtype="pdf",
                filename=report_path.name,
            )

    with smtplib.SMTP(smtp_server, int(smtp_port)) as smtp:
        smtp.starttls()
        smtp.login(smtp_username, smtp_password)
        smtp.send_message(message)

    return True, f"Report emailed to {recipient}."


def email_status():
    required_keys = [
        "FRACTUREAI_SMTP_SERVER",
        "FRACTUREAI_SMTP_PORT",
        "FRACTUREAI_SMTP_USERNAME",
        "FRACTUREAI_SMTP_PASSWORD",
        "FRACTUREAI_SENDER_EMAIL",
    ]
    missing = [key for key in required_keys if not os.getenv(key)]
    return {
        "configured": len(missing) == 0,
        "missing": missing,
    }


def dataset_stats():
    return {
        "train_fractured": len(list((DATASET_DIR / "train" / "fractured").glob("*"))) if (DATASET_DIR / "train" / "fractured").exists() else 0,
        "train_normal": len(list((DATASET_DIR / "train" / "non_fractured").glob("*"))) if (DATASET_DIR / "train" / "non_fractured").exists() else 0,
        "val_fractured": len(list((DATASET_DIR / "val" / "fractured").glob("*"))) if (DATASET_DIR / "val" / "fractured").exists() else 0,
        "val_normal": len(list((DATASET_DIR / "val" / "non_fractured").glob("*"))) if (DATASET_DIR / "val" / "non_fractured").exists() else 0,
        "test_fractured": len(list((DATASET_DIR / "test" / "fractured").glob("*"))) if (DATASET_DIR / "test" / "fractured").exists() else 0,
        "test_normal": len(list((DATASET_DIR / "test" / "non_fractured").glob("*"))) if (DATASET_DIR / "test" / "non_fractured").exists() else 0,
    }


def compute_performance_metrics():
    if not MODEL_PATH.exists():
        return None

    from tensorflow.keras.models import load_model

    _, _, test_generator = create_data_generators(DATASET_DIR)
    model = load_model(MODEL_PATH)
    predictions = model.predict(test_generator, verbose=0)
    predicted_labels = (predictions.flatten() >= 0.5).astype(int)
    true_labels = test_generator.classes

    tp = int(np.sum((predicted_labels == 1) & (true_labels == 1)))
    tn = int(np.sum((predicted_labels == 0) & (true_labels == 0)))
    fp = int(np.sum((predicted_labels == 1) & (true_labels == 0)))
    fn = int(np.sum((predicted_labels == 0) & (true_labels == 1)))
    accuracy = round(((tp + tn) / max(len(true_labels), 1)) * 100, 2)

    return {
        "accuracy": accuracy,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def base_context():
    search_query = request.args.get("q", "").strip()
    stats = dashboard_stats()
    history = list_cases(search_query)
    return {
        "dataset_ready": dataset_is_ready(),
        "model_ready": MODEL_PATH.exists(),
        "stats": stats,
        "history_cases": history,
        "recent_cases": recent_cases(),
        "search_query": search_query,
        "patient_details": patient_details_defaults(),
        "health_flags": [],
        "email_status": email_status(),
        "current_user": {
            "username": session.get("username"),
            "role": session.get("role"),
        },
    }


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = verify_user(username, password)

        if user:
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["role"] = user["role"]
            return redirect(url_for("index"))
        error = "Invalid username or password."

    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    context = base_context()

    if request.method == "POST":
        patient_details = build_patient_details(request.form)
        context["patient_details"] = patient_details_defaults(patient_details)

        if not MODEL_PATH.exists():
            context["error"] = "Model not found. Train the model first to create bone_fracture_model.h5."
            return render_template("index.html", **context)

        uploaded_file = request.files.get("xray_image")
        if uploaded_file is None or uploaded_file.filename == "":
            context["error"] = "Please choose an X-ray image."
            return render_template("index.html", **context)

        if not allowed_file(uploaded_file.filename):
            context["error"] = "Please upload a PNG, JPG, or JPEG image."
            return render_template("index.html", **context)

        safe_name = secure_filename(uploaded_file.filename)
        unique_name = f"{uuid4().hex}_{safe_name}"
        upload_path = UPLOAD_DIR / unique_name
        uploaded_file.save(upload_path)

        try:
            result, confidence = predict_fracture(upload_path)
            severity = estimate_severity(result, confidence)
            region_label = estimate_region_label(upload_path.name)
            health_score, overall_health, health_flags = compute_health_assessment(
                patient_details,
                result,
                confidence,
            )

            try:
                gradcam_path = generate_gradcam(upload_path)
                gradcam_error = None
                gradcam_name = gradcam_path.name
            except Exception as gradcam_exc:
                gradcam_error = f"Grad-CAM unavailable: {gradcam_exc}"
                gradcam_name = ""

            case_payload = {
                **patient_details,
                "result": result,
                "confidence": float(confidence),
                "severity": severity,
                "region_label": region_label,
                "health_score": health_score,
                "overall_health": overall_health,
                "uploaded_image": upload_path.name,
                "gradcam_image": gradcam_name,
                "report_file": "",
                "feedback": "",
                "created_at": datetime.now().strftime("%d %B %Y, %I:%M %p"),
            }

            report_name = create_pdf_report(case_payload)
            case_payload["report_file"] = report_name
            case_id = create_case(case_payload)

            context.update(
                {
                    "result": result,
                    "confidence": round(confidence, 2),
                    "severity": severity,
                    "region_label": region_label,
                    "health_score": health_score,
                    "overall_health": overall_health,
                    "health_flags": health_flags,
                    "uploaded_image": upload_path.name,
                    "gradcam_image": gradcam_name or None,
                    "gradcam_error": gradcam_error,
                    "report_file": report_name,
                    "case_id": case_id,
                    "stats": dashboard_stats(),
                    "history_cases": list_cases(request.args.get("q", "").strip()),
                    "recent_cases": recent_cases(),
                }
            )
            return render_template("index.html", **context)
        except Exception as exc:
            context["error"] = f"Prediction failed: {exc}"
            return render_template("index.html", **context)

    return render_template("index.html", **context)


@app.route("/feedback/<int:case_id>", methods=["POST"])
@login_required
def feedback(case_id):
    feedback_value = request.form.get("feedback", "").strip()
    if feedback_value in {"correct", "incorrect"}:
        update_feedback(case_id, feedback_value)
    return redirect(url_for("index"))


@app.route("/send-email/<int:case_id>", methods=["POST"])
@login_required
def send_email(case_id):
    case_record = get_case(case_id)
    context = base_context()

    if case_record is None:
        context["error"] = "Case not found."
        return render_template("index.html", **context)

    recipient_email = request.form.get("recipient_email", "").strip()
    success, message = send_report_email(case_record, recipient_email)
    context["message"] = message
    context["patient_details"] = patient_details_defaults(case_record)
    context["result"] = case_record["result"]
    context["confidence"] = round(case_record["confidence"], 2)
    context["severity"] = case_record["severity"]
    context["region_label"] = case_record["region_label"]
    context["health_score"] = round(case_record["health_score"], 2)
    context["overall_health"] = case_record["overall_health"]
    context["health_flags"] = []
    context["uploaded_image"] = case_record["uploaded_image"]
    context["gradcam_image"] = case_record["gradcam_image"] or None
    context["report_file"] = case_record["report_file"]
    context["case_id"] = case_record["id"]
    context["recipient_email"] = recipient_email or case_record.get("patient_email", "")
    if not success:
        context["error"] = message
    return render_template("index.html", **context)


@app.route("/performance")
@login_required
def performance():
    metrics = compute_performance_metrics()
    return render_template(
        "performance.html",
        metrics=metrics,
        dataset_counts=dataset_stats(),
        training_graph_exists=TRAINING_GRAPH_PATH.exists(),
        current_user={"username": session.get("username"), "role": session.get("role")},
    )


@app.route("/health-report/<int:case_id>")
@login_required
def health_report(case_id):
    case_record = get_case(case_id)
    if case_record is None:
        return redirect(url_for("index"))

    return render_template(
        "health_report.html",
        case_record=case_record,
        email_status=email_status(),
        current_user={"username": session.get("username"), "role": session.get("role")},
    )


@app.route("/uploads/<path:filename>")
@login_required
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/gradcam/<path:filename>")
@login_required
def gradcam_file(filename):
    return send_from_directory(str(GRADCAM_DIR), filename)


@app.route("/reports/<path:filename>")
@login_required
def report_file(filename):
    return send_from_directory(str(PDF_DIR), filename, as_attachment=True)


@app.route("/model-assets/<path:filename>")
@login_required
def model_assets(filename):
    return send_from_directory(str(MODEL_DIR), filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
