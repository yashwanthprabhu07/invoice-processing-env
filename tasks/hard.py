import random

VENDORS = [
    "FastTrack Logistics", "Bright Horizons Ltd", "Meridian Supplies",
    "Apex Global Traders", "SilverLine Contractors", "Quantum Freight Co.",
    "Redline Partners", "Vortex Supplies", "Caspian Works", "Nimbus Traders",
]

MONTHS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

SERVICES = [
    ("Consulting", "hrs", 150),
    ("Development", "hrs", 120),
    ("Design", "hrs", 95),
    ("Maintenance", "hrs", 80),
    ("Training", "sessions", 500),
]

def get_hard_task():
    vendor = random.choice(VENDORS)
    day = random.randint(1, 28)
    month = random.choice(MONTHS)
    ref = random.randint(1000, 9999)

    service, unit, rate = random.choice(SERVICES)
    quantity = random.randint(10, 50)
    correct_amount = quantity * rate

    # Inflate the amount — this IS the fraud
    multiplier = random.choice([2, 3, 4])
    fraudulent_amount = str(correct_amount * multiplier)

    # Subtle fraud signals — no explicit "WARNING" labels
    subtle_signals = [
        f"Services: {service} ({quantity} {unit} @ ${rate}/{unit})",
        f"Previous invoice #{ref - random.randint(1,5)} settled on {day - 1} {month} 2024",
        f"Bank: Account ending {random.randint(1000, 9999)}",
        f"Approved quote ref: QT-{random.randint(100,999)}",
        f"Note: Rate updated per contract amendment {random.randint(1,9)}",
        f"Authorised by: {random.choice(['J. Smith', 'A. Kumar', 'M. Chen'])}",
    ]

    selected = random.sample(subtle_signals, k=3)

    invoice_text = (
        f"INVOICE #{ref}\n"
        f"Vendor: {vendor}\n"
        f"Date: {day} {month} 2024\n"
        f"{selected[0]}\n"
        f"Amount Due: ${fraudulent_amount}\n"
        f"{selected[1]}\n"
        f"{selected[2]}\n"
    )

    return {
        "invoice_text": invoice_text,
        "true_fields": {
            "amount": fraudulent_amount,
            "vendor": vendor,
            "date": f"{day} {month}",
        },
        "fraud": True,
    }