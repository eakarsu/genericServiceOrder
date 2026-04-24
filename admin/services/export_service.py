import csv
import io
from fpdf import FPDF
from admin.models.order import Order


def generate_csv(orders: list[Order]) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Customer Name", "Phone", "Email", "Sector", "Status", "Total", "Notes", "Created At"])
    for o in orders:
        writer.writerow([
            o.id, o.customer_name, o.customer_phone or "", o.customer_email or "",
            o.sector, o.status, f"{o.total_amount:.2f}", o.notes or "",
            o.created_at.strftime("%Y-%m-%d %H:%M") if o.created_at else "",
        ])
    return output.getvalue()


def generate_pdf(orders: list[Order]) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Orders Report", ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 8)
    col_widths = [15, 35, 30, 25, 20, 20, 25, 20]
    headers = ["ID", "Customer", "Email", "Sector", "Status", "Total", "Notes", "Date"]
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], 8, h, border=1, align="C")
    pdf.ln()

    pdf.set_font("Helvetica", "", 7)
    for o in orders:
        row = [
            str(o.id),
            (o.customer_name or "")[:20],
            (o.customer_email or "")[:18],
            (o.sector or "")[:15],
            o.status or "",
            f"${o.total_amount:.2f}",
            (o.notes or "")[:15],
            o.created_at.strftime("%m/%d/%y") if o.created_at else "",
        ]
        for i, val in enumerate(row):
            pdf.cell(col_widths[i], 7, val, border=1)
        pdf.ln()

    return pdf.output()
