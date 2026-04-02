import random

VENDORS = [
    "ABC Ltd", "XYZ Corp", "Delta Inc", "Sunrise LLC",
    "Peak Solutions", "BlueStar Co", "Granite Partners",
    "Vega Systems", "Ironclad Inc", "Maple Exports",
]

AMOUNTS = [
    "1200", "1500", "2000", "3200", "4500",
    "5000", "6800", "7800", "9200", "10500",
]

MONTHS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

def get_easy_task():
    vendor = random.choice(VENDORS)
    amount = random.choice(AMOUNTS)
    day = random.randint(1, 28)
    month = random.choice(MONTHS)

    invoice_text = (
        f"Invoice from {vendor}. "
        f"Amount: ${amount}. "
        f"Date: {day} {month} 2024."
    )
    return {
        "invoice_text": invoice_text,
        "true_fields": {
            "amount": amount,
            "vendor": vendor,
            "date": f"{day} {month}",
        },
        "fraud": False,
    }